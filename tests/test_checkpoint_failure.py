import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from utils.trainer import SBIRTrainer


class ShortTrainer(SBIRTrainer):
    def train_epoch(self):
        return 1.

    def validate_epoch(self):
        self.last_metrics = {'Acc@1': .1, 'mAP': .2}
        return [.1, .2, .3]


class CheckpointFailureTests(unittest.TestCase):
    def make_trainer(self, root):
        return ShortTrainer(model=torch.nn.Linear(2, 2), train_loader=None,
                            test_loader=None, device='cpu', check_point=str(root / 'state.pth'),
                            logger=None, save_str='test', learning_rate=.001,
                            weight_decay=0., max_epochs=2, task='cl')

    def test_replace_failure_continues_epochs_and_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            with patch('utils.trainer.os.replace', side_effect=OSError(5, 'Input/output error')):
                trainer.train()
            self.assertEqual(len(trainer.train_losses), 2)
            self.assertTrue(Path(trainer.check_point + '.tmp').exists())
            self.assertTrue(trainer.save_checkpoint())
            saved = torch.load(trainer.check_point, weights_only=True)
            self.assertEqual(saved['epoch'], 2)

    def test_write_and_warning_failure_do_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            trainer.logger = Mock()
            trainer.logger.warning.side_effect = OSError('log filesystem unavailable')
            with patch('utils.trainer.torch.save', side_effect=RuntimeError('write failed')):
                self.assertFalse(trainer.save_checkpoint(True))
            trainer.logger.warning.assert_called_once()

    def test_best_publish_failure_preserves_previous_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp))
            self.assertTrue(trainer.save_checkpoint(True))
            previous = Path(trainer.check_point_best).read_bytes()
            trainer.current_epoch = 1
            import os
            replace = os.replace

            def fail_best(src, dst):
                if dst == trainer.check_point_best:
                    raise OSError(5, 'Input/output error')
                return replace(src, dst)

            with patch('utils.trainer.os.replace', side_effect=fail_best):
                self.assertFalse(trainer.save_checkpoint(True))
            self.assertEqual(Path(trainer.check_point_best).read_bytes(), previous)
            self.assertTrue(Path(trainer.check_point_best + '.tmp').exists())
            self.assertEqual(torch.load(trainer.check_point, weights_only=True)['epoch'], 2)


if __name__ == '__main__':
    unittest.main()
