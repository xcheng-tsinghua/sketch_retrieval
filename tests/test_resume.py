import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from utils.trainer import SBIRTrainer


class RandomTrainer(SBIRTrainer):
    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        x = torch.rand(4, 2) + random.random() + float(np.random.rand())
        loss = self.model(x).square().mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def validate_epoch(self):
        self.last_metrics = {'Acc@1': .1, 'mAP': .2}
        return [.1, .1, .1]


class ResumeTests(unittest.TestCase):
    def make_trainer(self, path, epochs):
        return RandomTrainer(torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.3)),
                             None, None, 'cpu', str(path), None, 'test', .001, 0., epochs, 'fg')

    def seed(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)

    def test_continuous_equals_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.seed()
            continuous = self.make_trainer(Path(tmp) / 'continuous.pth', 4)
            continuous.train()
            self.seed()
            first = self.make_trainer(Path(tmp) / 'resume.pth', 2)
            first.swanlab_state = {'id': 'saved-id', 'project': 'saved-project'}
            first.training_args = {'seed': 17}
            first.train()
            resumed = self.make_trainer(Path(tmp) / 'resume.pth', 4)
            self.assertTrue(resumed.load_checkpoint(resumed.check_point, True))
            self.assertEqual(resumed.swanlab_state, first.swanlab_state)
            self.assertEqual(resumed.training_args, first.training_args)
            # Initialization of external loggers must not alter resumed sampling.
            random.random(), np.random.rand(), torch.rand(5)
            resumed.swanlab_run = Mock()
            resumed.train()
            self.assertEqual(continuous.train_losses, resumed.train_losses)
            for key, tensor in continuous.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed.model.state_dict()[key]))
            self.assertEqual(continuous.scheduler.state_dict(), resumed.scheduler.state_dict())
            self.assertEqual([c.kwargs['step'] for c in resumed.swanlab_run.log.call_args_list], [3, 4])
            saved = torch.load(resumed.check_point, weights_only=True)
            self.assertEqual(saved['epoch'], 4)

    def test_invalid_training_state_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp) / 'invalid.pth', 1)
            torch.save({'model_state_dict': trainer.model.state_dict()}, trainer.check_point)
            with self.assertRaises(KeyError):
                trainer.load_checkpoint(trainer.check_point, True)

    def test_early_stop_state_is_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self.make_trainer(Path(tmp) / 'stopped.pth', 2)
            trainer.early_stop = ('Acc@1', .05)
            trainer.train()
            resumed = self.make_trainer(Path(tmp) / 'stopped.pth', 4)
            resumed.load_checkpoint(resumed.check_point, True)
            resumed.train_epoch = Mock()
            resumed.train()
            resumed.train_epoch.assert_not_called()
            self.assertEqual(resumed.early_stop, ('Acc@1', .05))


if __name__ == '__main__':
    unittest.main()
