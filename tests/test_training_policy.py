import unittest
from types import SimpleNamespace

from utils.training_policy import early_stop_policy, mix_training_split


class TrainingPolicyTests(unittest.TestCase):
    def test_fixed_size_reproducibility_and_unchanged_test(self):
        p = SimpleNamespace(
            train_pairs=[(f'train{i}', f'photo{i}', 'seen') for i in range(10)],
            test_pairs=[(f'test{i}', f'target{i}', 'held') for i in range(4)],
            train_images=[(f'photo{i}', 'seen') for i in range(10)],
            test_images=[(f'target{i}', 'held') for i in range(4)])
        self.assertIs(mix_training_split(p), p)
        for ratio in (.25, .4, 1.):
            with self.subTest(ratio=ratio):
                mixed = mix_training_split(p, ratio, 7)
                self.assertEqual(len(mixed.train_pairs), 10)
                self.assertEqual(sum(r in p.test_pairs for r in mixed.train_pairs), int(10 * ratio))
                self.assertEqual(mixed.train_pairs, mix_training_split(p, ratio, 7).train_pairs)
                self.assertEqual(mixed.test_pairs, p.test_pairs)
                self.assertEqual(mixed.test_images, p.test_images)
                self.assertTrue(all((r[1], r[2]) in mixed.train_images for r in mixed.train_pairs))
        self.assertEqual(len(p.train_pairs), 10)
        self.assertEqual(len(p.train_images), 10)
        for invalid in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                mix_training_split(p, invalid)

    def test_unpaired_cl_gallery(self):
        p = SimpleNamespace(train_pairs=[('s', 'p', 'a')], test_pairs=[('t', None, 'b')],
                            train_images=[('p', 'a')], test_images=[('q', 'b'), ('r', 'c')])
        self.assertEqual(mix_training_split(p, 1.).train_images, [('p', 'a'), ('q', 'b')])

    def test_all_thresholds_and_task_gating(self):
        models = ('sdgraph', 'gat', 'gcn', 'bidir_gru', 'gru', 'bidir_lstm', 'lstm')
        cases = [
            ('SKETCHY', 'cl', True, 'mAP@200', [.763, .508, .493, .455, .409, .471, .393]),
            ('sketchy', 'fg', True, 'Acc@1', [.328, .108, .174, .162, .141, .156, .135]),
            ('QMUL/Chair', 'fg', False, 'Acc@1', [.724, .251, .254, .238, .235, .232, .220]),
            ('qmul/shoe', 'fg', False, 'Acc@1', [.455, .190, .179, .176, .164, .158, .153]),
        ]
        for root, task, zs, metric, thresholds in cases:
            for model, threshold in zip(models, thresholds):
                self.assertEqual(early_stop_policy(root, task, zs, model), (metric, threshold))
            self.assertIsNone(early_stop_policy(root, task, not zs, 'sdgraph'))
            self.assertIsNone(early_stop_policy(root, task, zs, 'vit'))
        self.assertIsNone(early_stop_policy('other', 'fg', False, 'sdgraph'))


if __name__ == '__main__':
    unittest.main()
