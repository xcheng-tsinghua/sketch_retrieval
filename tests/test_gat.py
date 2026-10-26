import unittest

import torch
import torch.nn.functional as F

import options
from encoders.gat import GATSketchEncoder, GraphAttentionLayer
from encoders.create_sketch_encoder import create_sketch_encoder


class GATTests(unittest.TestCase):
    def test_attention_matches_paper_equation(self):
        torch.manual_seed(4)
        layer = GraphAttentionLayer(3, 2, heads=2, dropout=0.).double()
        x = torch.randn(1, 3, 3, dtype=torch.double, requires_grad=True)
        adjacency = torch.tensor([[[1, 1, 0], [1, 1, 1], [0, 1, 1]]], dtype=torch.bool)
        valid = torch.ones(1, 3, dtype=torch.bool)
        actual = layer(x, adjacency, valid)
        # Independent scalar implementation of Eq. (3)--(5), using concatenation.
        expected_heads = []
        for head in range(2):
            weight = layer.projection.weight[head * 2:(head + 1) * 2]
            h = F.linear(x[0], weight)
            a = torch.cat((layer.attn_source[head], layer.attn_neighbor[head]))
            rows = []
            for i in range(3):
                neighbors = adjacency[0, i].nonzero().flatten()
                scores = torch.stack([F.leaky_relu(a @ torch.cat((h[i], h[j])), .2) for j in neighbors])
                rows.append((scores.softmax(0)[:, None] * h[neighbors]).sum(0))
            expected_heads.append(torch.stack(rows))
        torch.testing.assert_close(actual[0], torch.cat(expected_heads, dim=-1))
        actual.square().sum().backward()
        for parameter in layer.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        layer.concat = False
        torch.testing.assert_close(layer(x, adjacency, valid)[0], torch.stack(expected_heads).mean(0))

    def test_attention_permutation_equivariance_and_mask(self):
        layer = GraphAttentionLayer(3, 2, heads=2, dropout=0.).eval()
        x = torch.randn(1, 4, 3)
        valid = torch.tensor([[True, True, True, False]])
        adjacency = torch.ones(1, 4, 4, dtype=torch.bool)
        reference = layer(x, adjacency, valid)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = layer(x[:, permutation], adjacency[:, permutation][:, :, permutation], valid[:, permutation])
        torch.testing.assert_close(permuted, reference[:, permutation])
        x[:, 3] = 1000.
        torch.testing.assert_close(layer(x, adjacency, valid), reference)
        self.assertEqual(reference[:, 3].abs().sum().item(), 0.)

    @staticmethod
    def sketch():
        # Two strokes, third point is a real zero-coordinate point.
        return torch.tensor([[[0., 0., 0., 1., 0.], [1., 0., 1., 0., 0.],
                              [-1., 0., 0., 1., 0.], [0., 1., 1., 0., 1.]]])

    def test_stroke_edges_and_padding_invariance(self):
        encoder = GATSketchEncoder(16, neighbors=0, dropout=0.).eval()
        sketch = self.sketch()
        _, adjacency, valid = encoder.build_graph(sketch)
        self.assertTrue(valid.all())
        self.assertTrue(adjacency[0, 0, 1])
        self.assertFalse(adjacency[0, 1, 2])
        self.assertTrue(adjacency[0, 2, 3])
        padded = F.pad(sketch, (0, 0, 0, 6))
        padded[:, 4:, :2] = 500.  # Padding coordinates must never influence geometry.
        for neighbors in (0, 2, 8):
            encoder.neighbors = neighbors
            torch.testing.assert_close(encoder(sketch), encoder(padded), atol=1e-6, rtol=1e-5)
        self.assertEqual(encoder(padded).shape, (1, 16))
        torch.testing.assert_close(encoder(padded).norm(dim=1), torch.ones(1))

    def test_single_point_empty_and_factory(self):
        args = options.parse_args(['--sketch_model', 'gat'])
        info = options.get_encoder_info(args.sketch_model)
        encoder = create_sketch_encoder('gat', output_dim=16, freeze_backbone=True,
                                         sketch_format=info['sketch_format'])
        self.assertFalse(encoder.gat1.projection.weight.requires_grad)
        self.assertTrue(encoder.readout[0].weight.requires_grad)
        single = torch.tensor([[[0., 0., 1., 0., 1.]]])
        self.assertTrue(torch.isfinite(encoder(single)).all())
        with self.assertRaisesRegex(ValueError, 'no valid points'):
            encoder(torch.zeros(1, 5, 5))


if __name__ == '__main__':
    unittest.main()
