import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import options
from PIL import Image

from data.retrieval_datasets import create_sketch_image_dataloaders
from data.shared_cache import SharedTensorCache


class SharedCacheTests(unittest.TestCase):
    def test_model_and_mixed_split_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            photos, rows = [], []
            for i in range(3):
                photo, sketch = Path(tmp) / f'{i}.png', Path(tmp) / f'{i}.txt'
                Image.new('RGB', (8, 8), (12, 34, 56)).save(photo)
                sketch.write_text('0 0 0\n1 1 1\n')
                photos.append((str(photo), 'a'))
                rows.append((str(sketch), str(photo), 'a'))
            full = SimpleNamespace(train_pairs=rows[:2], test_pairs=rows[2:],
                                   train_images=photos, test_images=photos,
                                   common_categories=['a'], is_zero_shot=False)
            mixed = SimpleNamespace(**vars(full))
            mixed.train_pairs = [rows[0], rows[2]]
            kwargs = dict(batch_size=2, num_workers=0, pre_load=mixed,
                          is_full_train=False, task='fg', cache_preload=full,
                          cache_dir=str(Path(tmp) / 'cache'))
            first, _ = create_sketch_image_dataloaders(
                **kwargs, sketch_format=options.get_encoder_info('gat')['sketch_format'])
            mixed.train_pairs = [rows[1], rows[2]]
            with patch('data.retrieval_datasets.utils.image_loader', side_effect=AssertionError('cache miss')), \
                 patch('data.retrieval_datasets.utils.s3_file_to_s5', side_effect=AssertionError('cache miss')):
                second, _ = create_sketch_image_dataloaders(
                    **kwargs, sketch_format=options.get_encoder_info('gcn')['sketch_format'])
            torch.testing.assert_close(first.dataset.sketch_loader.storage,
                                       second.dataset.sketch_loader.storage)
            self.assertEqual(len(list((Path(tmp) / 'cache').glob('*.pt'))), 2)

    def test_persistent_hit_invalidation_and_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.txt'
            source.write_text('original')
            loader = Mock(return_value=torch.tensor([[1., 2.]]))
            kwargs = dict(cache_dir=str(Path(tmp) / 'cache'), signature={'fmt': 'sequence'})
            first = SharedTensorCache([str(source)], loader, **kwargs)
            loader.reset_mock()
            second = SharedTensorCache([str(source)], loader, **kwargs)
            loader.assert_not_called()
            self.assertTrue(second.storage.is_shared())
            torch.testing.assert_close(first(str(source)), second(str(source)))
            source.write_text('changed size')
            SharedTensorCache([str(source)], loader, **kwargs)
            self.assertEqual(loader.call_count, 1)
            SharedTensorCache([str(source)], loader, **dict(kwargs, signature={'fmt': 'other'}))
            self.assertEqual(loader.call_count, 2)
            SharedTensorCache([str(source)], loader, **kwargs, rebuild=True)
            self.assertEqual(loader.call_count, 3)
            self.assertFalse(list((Path(tmp) / 'cache').glob('*.tmp')))

    def test_formats_match_and_workers_need_no_source_files(self):
        for fmt in ('img', 's3 -> img', 's3 -> s5, max_length: 4',
                    'stk, n_stk: 2, n_stk_pnt: 2'):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                photos, sketches = [], []
                for i in range(3):
                    photo = root / f'photo{i}.png'
                    Image.new('RGB', (9, 11), (i * 50, 89, 173)).save(photo)
                    sketch = root / f'sketch{i}.txt'
                    if fmt == 'img':
                        sketch = root / f'sketch{i}.png'
                        Image.new('RGB', (7, 9), (21, i * 30, 240)).save(sketch)
                    elif fmt.startswith('stk'):
                        sketch.write_text('0 0\n1 1\n2 1\n3 0\n')
                    else:
                        sketch.write_text('0 0 0\n1 1 1\n2 1 0\n3 0 1\n')
                    photos.append((str(photo), 'a'))
                    sketches.append((str(sketch), str(photo), 'a'))
                preload = SimpleNamespace(train_pairs=sketches[:2], test_pairs=sketches[1:],
                                          train_images=photos, test_images=photos,
                                          common_categories=['a'], is_zero_shot=False)
                args = (2, 0, preload, 'fmt: ' + fmt, False, 'fg')
                cached, test = create_sketch_image_dataloaders(*args, n_neg=1)
                direct, _ = create_sketch_image_dataloaders(*args, n_neg=1, cache_mode='none')
                ds = cached.dataset
                self.assertIs(ds.image_loader, test.dataset.image_loader)
                self.assertIs(ds.sketch_loader, test.dataset.sketch_loader)
                self.assertEqual(len(ds.sketch_loader.indices), 3)
                self.assertTrue(ds.sketch_loader.storage.is_shared())
                self.assertTrue(ds.image_loader.storage.is_shared())
                for path, _ in photos:
                    torch.testing.assert_close(ds.image_loader(path), direct.dataset.image_loader(path),
                                               rtol=0, atol=0)
                for path, _, _ in sketches:
                    expected = direct.dataset.sketch_loader(path)
                    torch.testing.assert_close(ds.sketch_loader(path), expected, rtol=0, atol=0)
                    ds.sketch_loader(path).zero_()
                    torch.testing.assert_close(ds.sketch_loader(path), expected, rtol=0, atol=0)
                for path in root.iterdir():
                    path.unlink()
                # Explicit spawn tests Windows/Linux worker serialization of shared storage.
                loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=1,
                                                    multiprocessing_context='spawn')
                self.assertEqual(next(iter(loader))[0].shape[0], 2)
                for mode in ('sketch', 'image'):
                    ds.back_mode = mode
                    self.assertEqual(next(iter(loader)).shape[0], 2)


if __name__ == '__main__':
    unittest.main()
