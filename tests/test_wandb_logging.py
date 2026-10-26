import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils.wandb_logging import get_project_name, init_wandb


class WandbLoggingTests(unittest.TestCase):
    def args(self, **kwargs):
        values = dict(task='fg', is_zero_shot='False', local='True',
                      root_local='D:/data/qmul_v2_fit/chair',
                      root_sever='/data/sketchy', wandb_project='', seed=42)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_project_names(self):
        for task in ('fg', 'cl'):
            for zero_shot in ('True', 'False'):
                for path, dataset in [('D:/SKETCHY', 'sketchy'),
                                      (r'D:\qmul_v2_fit\chair', 'qmul_chair'),
                                      ('/data/qmul_shoe', 'qmul_shoe')]:
                    args = self.args(task=task, is_zero_shot=zero_shot, root_local=path)
                    mode = 'ZS_SBIR' if zero_shot == 'True' else 'SBIR'
                    self.assertEqual(get_project_name(args), f'{task.upper()}_{mode}_{dataset}')
        self.assertEqual(get_project_name(self.args(local='False')), 'FG_SBIR_sketchy')
        self.assertEqual(get_project_name(self.args(wandb_project='custom')), 'custom')
        with self.assertRaises(ValueError):
            get_project_name(self.args(root_local='/data/unknown'))

    def test_login_and_config(self):
        wandb = Mock()
        args = self.args()
        with patch.dict('sys.modules', {'wandb': wandb}), \
                patch('dotenv.dotenv_values', return_value={'WANDB_API_KEY': 'test-key'}):
            self.assertIs(init_wandb(args, 'model_42'), wandb.init.return_value)
        wandb.login.assert_called_once_with(key='test-key', relogin=True)
        wandb.init.assert_called_once_with(project='FG_SBIR_qmul_chair',
                                          name='model_42', config=vars(args))
        self.assertNotIn('WANDB_API_KEY', wandb.init.call_args.kwargs['config'])

    def test_missing_key(self):
        wandb = Mock()
        with patch.dict('sys.modules', {'wandb': wandb}), \
                patch('dotenv.dotenv_values', return_value={}):
            with self.assertRaisesRegex(ValueError, 'WANDB_API_KEY'):
                init_wandb(self.args(), 'model_42')
        wandb.login.assert_not_called()


if __name__ == '__main__':
    unittest.main()
