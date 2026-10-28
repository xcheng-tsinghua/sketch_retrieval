"""SwanLab setup without importing the training/model dependencies."""
from pathlib import Path


def get_project_name(args):
    if args.swanlab_project:
        return args.swanlab_project
    root = args.root_local if str(args.local).lower() == 'true' else args.root_sever
    root = str(root).lower().replace('\\', '/')
    matches = [name for name in ('sketchy', 'qmul_chair', 'qmul_shoe')
               if name in root or (name.startswith('qmul_') and 'qmul' in root
                                   and name.split('_')[1] in root)]
    if len(matches) != 1:
        raise ValueError('Cannot uniquely identify sketchy, qmul_chair or qmul_shoe '
                         'from dataset root; specify --swanlab_project explicitly')
    mode = 'ZS_SBIR' if str(args.is_zero_shot).lower() == 'true' else 'SBIR'
    return f'{args.task.upper()}_{mode}_{matches[0]}'


def init_swanlab(args, save_str, resume_state=None):
    import swanlab
    from dotenv import dotenv_values

    project = resume_state['project'] if resume_state else get_project_name(args)
    env_path = Path(__file__).resolve().parents[1] / '.env'
    api_key = dotenv_values(env_path).get('SwanLab_API_KEY')
    if not api_key or not api_key.strip():
        raise ValueError('Set SwanLab_API_KEY in the project .env file before training')
    swanlab.login(api_key=api_key.strip(), save=False)
    resume_kwargs = {}
    if getattr(args, 'swanlab_workspace', ''):
        resume_kwargs['workspace'] = args.swanlab_workspace
    if resume_state:
        resume_kwargs = {'id': resume_state['id'], 'resume': 'must'}
        if resume_state.get('workspace'):
            resume_kwargs['workspace'] = resume_state['workspace']
    return swanlab.init(project=project, experiment_name=save_str, config=vars(args),
                        **resume_kwargs)
