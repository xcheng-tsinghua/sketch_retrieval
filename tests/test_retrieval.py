import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

import options
from data.retrieval_datasets import DatasetPreload, SketchImageDataset
from utils.loss_func import info_nce_multi_neg, info_nce_multi_positive
from utils.retrieval_metrics import evaluate_retrieval
from utils.trainer import SBIRTrainer


class RetrievalTests(unittest.TestCase):
    def test_category_vs_instance_and_small_gallery(self):
        args = (torch.tensor([[3., 2., 1.]]), torch.eye(3))
        cl, _ = evaluate_retrieval(*args, 'cl', [2], [1], [0, 1, 1], chunk_size=1)
        fg, _ = evaluate_retrieval(*args, 'fg', [2], [1], [0, 1, 1], chunk_size=2)
        self.assertEqual(cl['Acc@1'], 0.)
        self.assertEqual(cl['Acc@5'], 1.)
        self.assertAlmostEqual(cl['mAP'], 7 / 12, places=6)
        self.assertAlmostEqual(fg['mAP'], 1 / 3, places=6)
        self.assertAlmostEqual(cl['P@200'], 2 / 3, places=6)
        truncated, _ = evaluate_retrieval(*args, 'cl', [2], [1], [0, 1, 1], map_k=2)
        self.assertAlmostEqual(truncated['mAP@2'], .25)
        same_class, _ = evaluate_retrieval(*args, 'cl', [2], [1], [1, 0, 1])
        self.assertEqual(same_class['Acc@1'], 1.)  # A different instance is a valid CL hit.
        self.assertEqual(fg['Acc@1'], 0.)

    def test_multi_positive_loss(self):
        torch.manual_seed(7)
        q = torch.randn(3, 8, requires_grad=True)
        p = torch.randn(3, 2, 8, requires_grad=True)
        n = torch.randn(3, 4, 8, requires_grad=True)
        expected = info_nce_multi_neg(q, p[:, 0], n)
        actual = info_nce_multi_positive(q, p[:, :1], n)
        torch.testing.assert_close(expected, actual)
        loss = info_nce_multi_positive(q, p, n)
        loss.backward()
        for tensor in (q, p, n):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    @staticmethod
    def make_files(root, category, instance, split=''):
        for kind in ('sketch', 'photo'):
            folder = root / kind / split / category
            folder.mkdir(parents=True, exist_ok=True)
            names = [instance + '_1', instance + '_2'] if kind == 'sketch' else [instance]
            for name in names:
                Image.new('RGB', (8, 8), 'white').save(folder / (name + '.png'))

    def test_split_gallery_and_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cat in ('a', 'b', 'held'):
                for i in range(3):
                    self.make_files(root, cat, str(i))
                Image.new('RGB', (8, 8)).save(root / 'photo' / cat / 'distractor.png')
            kwargs = dict(sketch_root=root / 'sketch', image_root=root / 'photo',
                          sketch_suffix='png', image_suffix='png', is_multi_pair=True,
                          zs_test_classes=['held'], is_zero_shot=True)
            preload = DatasetPreload(**kwargs, task='cl')
            self.assertEqual(len(preload.test_images), 4)  # Unpaired photo retained.
            self.assertEqual({r[2] for r in preload.test_pairs}, {'held'})
            data = SketchImageDataset(True, preload, None, None, 'fmt: img', False,
                                      'cl', n_pos=2, n_neg=12)
            sk, pos, neg = data[0]
            self.assertEqual(pos.shape, (2, 3, 224, 224))
            self.assertEqual(neg.shape, (12, 3, 224, 224))
            kwargs['is_zero_shot'] = False
            one = DatasetPreload(**kwargs, task='fg')
            two = DatasetPreload(**kwargs, task='fg')
            self.assertEqual(one.train_pairs, two.train_pairs)
            self.assertFalse({r[1] for r in one.train_pairs} & {r[1] for r in one.test_pairs})
            self.assertFalse(set(one.train_images) & set(one.test_images))

    def test_predefined_disjoint_classes_and_unpaired_cl_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_files(root, 'seen', '0', 'train')
            self.make_files(root, 'unseen', '0', 'test')
            Image.new('RGB', (8, 8)).save(root / 'sketch' / 'test' / 'unseen' / 'unpaired.png')
            kwargs = dict(sketch_root=root / 'sketch', image_root=root / 'photo',
                          sketch_suffix='png', image_suffix='png', is_multi_pair=True)
            preload = DatasetPreload(**kwargs, task='cl')
            self.assertEqual(preload.common_categories, ['seen', 'unseen'])
            self.assertEqual(len(preload.test_pairs), 3)
            with self.assertRaisesRegex(ValueError, 'No paired photo'):
                DatasetPreload(**kwargs, task='fg')

    def test_yaml_defaults_and_cli_overrides(self):
        args = options.parse_args(['--is_load_config', 'True', '--configs_name',
                                   'sketchy_cl_zs.yaml', '--n_pos', '3', '--local', 'False'])
        self.assertEqual(args.n_pos, 3)
        self.assertEqual(args.local, 'False')
        self.assertEqual(args.image_suffix, 'jpg')
        self.assertEqual(args.task, 'cl')

    def test_qmul_style_repeated_gallery_excludes_other_split_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_files(root, 'class', 'train_id', 'train')
            self.make_files(root, 'class', 'test_id', 'test')
            for split, other in (('train', 'test_id'), ('test', 'train_id')):
                Image.new('RGB', (8, 8)).save(root / 'photo' / split / 'class' / (other + '.png'))
            p = DatasetPreload(root / 'sketch', root / 'photo', 'png', 'png',
                               is_multi_pair=True, is_zero_shot=False, task='fg')
            self.assertEqual([Path(r[0]).stem for r in p.train_images], ['train_id'])
            self.assertEqual([Path(r[0]).stem for r in p.test_images], ['test_id'])

    def test_resume_optimizer_scheduler_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            kwargs = dict(train_loader=None, test_loader=None, device='cpu',
                          check_point=str(Path(tmp) / 'state.pth'), logger=None,
                          save_str='test', learning_rate=.001, weight_decay=0.,
                          max_epochs=5, task='cl')
            trainer = SBIRTrainer(model=model, **kwargs)
            model(torch.ones(1, 2)).sum().backward()
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.current_epoch = 2
            trainer.best_acc = .6
            trainer.save_checkpoint(True)
            resumed = SBIRTrainer(model=torch.nn.Linear(2, 2), **kwargs)
            self.assertTrue(resumed.load_checkpoint(kwargs['check_point'], True))
            self.assertEqual(resumed.current_epoch, 3)
            self.assertEqual(resumed.best_acc, .6)
            self.assertTrue(resumed.optimizer.state)
            self.assertEqual(resumed.scheduler.state_dict(), trainer.scheduler.state_dict())

    def test_best_checkpoint_between_periodic_saves(self):
        class ShortTrainer(SBIRTrainer):
            def train_epoch(self):
                return 1.

            def validate_epoch(self):
                self.last_metrics = {'Acc@1': .1, 'mAP': [.8, .2][self.current_epoch]}
                return [.1, .2, .3]

        with tempfile.TemporaryDirectory() as tmp:
            trainer = ShortTrainer(torch.nn.Linear(2, 2), None, None, 'cpu',
                                   str(Path(tmp) / 'test.pth'), None, 'test', .001, 0., 2, 'cl')
            trainer.train()
            best = torch.load(trainer.check_point_best, weights_only=True)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(best['best_acc'], .8)

    def test_optional_extended_gallery_and_class_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cat in ('seen', 'unseen'):
                self.make_files(root, cat, '0')
                folder = root / 'extra' / cat
                folder.mkdir(parents=True)
                Image.new('RGB', (8, 8)).save(folder / 'extra.png')
            p = DatasetPreload(root / 'sketch', root / 'photo', 'png', 'png',
                               is_multi_pair=True, is_zero_shot=True, task='fg',
                               zs_test_classes=['unseen'], extended_image_root=root / 'extra')
            self.assertEqual(len(p.train_images), 2)
            self.assertEqual(len(p.test_images), 2)
            self.assertEqual({r[1] for r in p.test_images}, {'unseen'})
            self.assertTrue(all(Path(r[1]).stem == '0' for r in p.test_pairs))


if __name__ == '__main__':
    unittest.main()
