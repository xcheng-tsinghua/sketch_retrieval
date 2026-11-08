import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import options
import train


class CheckpointFallbackTests(unittest.TestCase):
    def test_checkpoint_argument_removed(self):
        self.assertFalse(hasattr(options.parse_args([]), 'checkpoint'))
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            options.parse_args(['--checkpoint', 'old.pth'])

    def test_missing_and_corrupt_checkpoint_warn_and_start_fresh(self):
        real_prepare = train.prepare_training
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as tmp:
                args = options.parse_args(['--is_load_ckpt', 'True', '--weight_dir', tmp])
                original = copy.deepcopy(vars(args))
                path = Path(train.utils.get_check_point(tmp, train.utils.get_save_str(args)))
                if corrupt:
                    path.write_bytes(b'not a torch checkpoint')
                fresh = Mock(save_str='fresh', swanlab_state=None)
                run = Mock(id='new-id')

                def prepare(current):
                    if current.is_load_ckpt == 'True':
                        return real_prepare(current)
                    self.assertEqual(vars(current), {**original, 'is_load_ckpt': 'False'})
                    return fresh, False, Mock()

                with patch.object(train, 'prepare_training', side_effect=prepare), \
                        patch.object(train, 'init_swanlab', return_value=run) as init, \
                        self.assertWarnsRegex(RuntimeWarning, 'Starting a new training run'):
                    train.main(args)
                init.assert_called_once()
                self.assertIsNone(init.call_args.args[2])
                fresh.train.assert_called_once()
                self.assertEqual(fresh.swanlab_state['id'], 'new-id')
                self.assertEqual(vars(args), original)

    def test_partial_restore_discards_checkpoint_arguments_and_trainer(self):
        args = options.parse_args(['--is_load_ckpt', 'True', '--seed', '17'])
        fresh = Mock(save_str='fresh', swanlab_state=None)
        calls = []

        def prepare(current):
            calls.append(current)
            if len(calls) == 1:
                current.seed = 999
                raise ValueError('optimizer state is invalid after model loading')
            self.assertEqual(current.seed, 17)
            self.assertEqual(current.is_load_ckpt, 'False')
            return fresh, False, Mock()

        with patch.object(train, 'prepare_training', side_effect=prepare), \
                patch.object(train, 'init_swanlab', return_value=Mock(id='new-id')), \
                self.assertWarns(RuntimeWarning):
            train.main(args)
        fresh.train.assert_called_once()

    def test_evaluation_failure_does_not_start_training(self):
        args = options.parse_args(['--eval_only'])
        with patch.object(train, 'prepare_training', side_effect=FileNotFoundError), \
                patch.object(train, 'init_swanlab') as init:
            with self.assertRaises(FileNotFoundError):
                train.main(args)
        init.assert_not_called()


if __name__ == '__main__':
    unittest.main()
