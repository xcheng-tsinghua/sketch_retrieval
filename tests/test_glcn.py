import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

import options
from encoders.glcn import GraphLearningLayer, GLCNSketchEncoder
from encoders.create_sketch_encoder import create_sketch_encoder
from encoders.sbir_model_wrapper import SBIRModelWrapper
from utils.trainer import SBIRTrainer


class GLCNTests(unittest.TestCase):
    def test_graph_and_regularizer_match_equations(self):
        layer = GraphLearningLayer(1, 1, gamma=.3).double()
        with torch.no_grad():
            layer.projection.weight.fill_(2.)
            layer.metric.fill_(.5)
        features = torch.tensor([[[0.], [1.], [3.]]], dtype=torch.double)
        adjacency = torch.tensor([[[1, 1, 0], [1, 1, 1], [0, 1, 1]]], dtype=torch.bool)
        valid = torch.ones(1, 3, dtype=torch.bool)
        graph, penalty = layer(features, adjacency, valid)
        expected = torch.zeros(3, 3, dtype=torch.double)
        coords = [0., 2., 6.]
        for i in range(3):
            neighbors = [j for j in range(3) if adjacency[0, i, j]]
            scores = torch.tensor([max(0., .5 * abs(coords[i] - coords[j])) for j in neighbors], dtype=torch.double)
            expected[i, neighbors] = scores.softmax(0)
        torch.testing.assert_close(graph[0], expected)
        loss = sum((coords[i] - coords[j]) ** 2 * expected[i, j] + .3 * expected[i, j] ** 2
                   for i in range(3) for j in range(3)) / 3
        torch.testing.assert_close(penalty, loss)
        torch.testing.assert_close(graph.sum(-1), torch.ones(1, 3, dtype=torch.double))
        penalty.backward()
        for parameter in layer.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    @staticmethod
    def sketch():
        return torch.tensor([[[0., 0., 0., 1., 0.], [1., 0., 1., 0., 0.],
                              [0., 1., 1., 0., 1.]]])

    def test_padding_gradients_and_eval_state(self):
        model = GLCNSketchEncoder(16, dropout=0.)
        sketch = self.sketch()
        features = model(sketch)
        penalty = model.auxiliary_loss()
        padded = F.pad(sketch, (0, 0, 0, 7))
        padded[:, 3:, :2] = 1000.
        torch.testing.assert_close(model(padded), features, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model.auxiliary_loss(), penalty)
        loss = (features * torch.randn_like(features)).sum() + penalty
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.graph_learning.projection.weight.grad.abs().sum().item(), 0.)
        model.eval()
        result = model(sketch)
        self.assertIsNone(model.auxiliary_loss())
        torch.testing.assert_close(result.norm(dim=1), torch.ones(1))

    def test_alias_freezing_and_single_node(self):
        for name in ('gcn', 'glcn'):
            args = options.parse_args(['--sketch_model', name])
            model = create_sketch_encoder(args.sketch_model, output_dim=16, freeze_backbone=True,
                                          sketch_format=options.get_encoder_info(name)['sketch_format'])
            self.assertIsInstance(model, GLCNSketchEncoder)
            self.assertFalse(model.graph_learning.metric.requires_grad)
            self.assertFalse(model.conv1.weight.requires_grad)
            self.assertTrue(model.readout[0].weight.requires_grad)
            self.assertTrue(torch.isfinite(model(torch.tensor([[[0., 0., 1., 0., 1.]]]))).all())

    def test_trainer_optimizes_graph_regularizer(self):
        class SmallWrapper(SBIRModelWrapper):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.sketch_encoder = GLCNSketchEncoder(16, dropout=0., graph_loss_weight=1.)
                self.sketch_projection = torch.nn.Identity()
                self.image_encoder = torch.nn.Identity()
                self.freeze_image_encoder = False
                self.image_projection = torch.nn.Linear(3, 16)
                self.logit_scale = torch.nn.Parameter(torch.zeros(()))

            def encode_image(self, images):
                return F.normalize(self.image_projection(images.mean(dim=(-1, -2))), dim=-1)

        dataset = torch.utils.data.TensorDataset(self.sketch(), torch.randn(1, 3, 2, 2),
                                                 torch.randn(1, 2, 3, 2, 2))
        dataset.back_train_data = lambda: None
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)
        model = SmallWrapper()
        before = model.sketch_encoder.graph_learning.projection.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SBIRTrainer(model, loader, None, 'cpu', str(Path(tmp) / 'test.pth'),
                                  None, 'test', .001, 0., 1, 'fg')
            # Suppress retrieval gradients: parameter changes must come from the GL penalty.
            trainer.criterion = lambda s, p, n: (s.sum() + p.sum() + n.sum()) * 0.
            self.assertGreater(trainer.train_epoch(), 0.)
        self.assertFalse(torch.equal(before, model.sketch_encoder.graph_learning.projection.weight))


if __name__ == '__main__':
    unittest.main()
