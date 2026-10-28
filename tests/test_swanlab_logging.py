import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils.swanlab_logging import get_project_name, init_swanlab


class SwanLabLoggingTests(unittest.TestCase):
    def args(self, **kwargs):
        values = dict(task='fg', is_zero_shot='False', local='True',
                      root_local='D:/data/qmul_v2_fit/chair',
                      root_sever='/data/sketchy', swanlab_project='', seed=42)
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
        self.assertEqual(get_project_name(self.args(swanlab_project='custom')), 'custom')
        with self.assertRaises(ValueError):
            get_project_name(self.args(root_local='/data/unknown'))

    def test_login_and_config(self):
        swanlab = Mock()
        args = self.args()
        with patch.dict('sys.modules', {'swanlab': swanlab}), \
                patch('dotenv.dotenv_values', return_value={'SwanLab_API_KEY': 'test-key'}):
            self.assertIs(init_swanlab(args, 'model_42'), swanlab.init.return_value)
        swanlab.login.assert_called_once_with(api_key='test-key', save=False)
        swanlab.init.assert_called_once_with(project='FG_SBIR_qmul_chair',
                                          experiment_name='model_42', config=vars(args))
        self.assertNotIn('SwanLab_API_KEY', swanlab.init.call_args.kwargs['config'])

    def test_resume_original_experiment(self):
        swanlab = Mock()
        state = {'id': 'original-id', 'project': 'original-project', 'workspace': 'team'}
        with patch.dict('sys.modules', {'swanlab': swanlab}), \
                patch('dotenv.dotenv_values', return_value={'SwanLab_API_KEY': 'test-key'}):
            init_swanlab(self.args(), 'model_42', state)
        kwargs = swanlab.init.call_args.kwargs
        self.assertEqual(kwargs['id'], 'original-id')
        self.assertEqual(kwargs['resume'], 'must')
        self.assertEqual(kwargs['project'], 'original-project')
        self.assertEqual(kwargs['workspace'], 'team')

    def test_missing_key(self):
        swanlab = Mock()
        with patch.dict('sys.modules', {'swanlab': swanlab}), \
                patch('dotenv.dotenv_values', return_value={}):
            with self.assertRaisesRegex(ValueError, 'SwanLab_API_KEY'):
                init_swanlab(self.args(), 'model_42')
        swanlab.login.assert_not_called()


if __name__ == '__main__':
    unittest.main()
