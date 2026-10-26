"""Fixed-size split mixing and task-specific stopping policies."""
import copy
import random


MODELS = ('sdgraph', 'gat', 'gcn', 'bidir_gru', 'gru', 'bidir_lstm', 'lstm')
THRESHOLDS = {
    'sketchy_cl': dict(zip(MODELS, (.763, .508, .493, .455, .409, .471, .393))),
    'sketchy_fg': dict(zip(MODELS, (.328, .108, .174, .162, .141, .156, .135))),
    'chair': dict(zip(MODELS, (.724, .251, .254, .238, .235, .232, .220))),
    'shoe': dict(zip(MODELS, (.455, .190, .179, .176, .164, .158, .153))),
}


def early_stop_policy(root, task, is_zero_shot, model):
    root = str(root).lower()
    key = None
    if 'sketchy' in root and is_zero_shot and task in ('cl', 'fg'):
        key = 'sketchy_' + task
    elif 'qmul' in root and not is_zero_shot and task == 'fg':
        key = next((name for name in ('chair', 'shoe') if name in root), None)
    threshold = THRESHOLDS.get(key, {}).get(model)
    if threshold is None:
        return None
    return ('mAP@200' if task == 'cl' else 'Acc@1', threshold)


def mix_training_split(preload, ratio=0., seed=42):
    """Replace floor(N * ratio) training queries; never modify the test split.

    Sample without replacement where possible, with replacement only if the
    requested test count exceeds the test pool. Keep original training gallery
    plus the paired test targets (or category gallery for unpaired CL queries).
    """
    if not 0 <= ratio <= 1:
        raise ValueError('test_mix_ratio must be between 0 and 1')
    if ratio == 0:
        return preload
    count = int(len(preload.train_pairs) * ratio)
    if count == 0:
        return preload
    if not preload.test_pairs:
        raise ValueError('Cannot mix an empty test split')
    rng = random.Random(seed)
    test_rows = (rng.sample(preload.test_pairs, count) if count <= len(preload.test_pairs)
                 else rng.choices(preload.test_pairs, k=count))
    mixed = copy.copy(preload)
    mixed.train_pairs = rng.sample(preload.train_pairs, len(preload.train_pairs) - count) + test_rows
    rng.shuffle(mixed.train_pairs)
    targets = {row[1] for row in test_rows if row[1] is not None}
    unpaired_categories = {row[2] for row in test_rows if row[1] is None}
    mixed.train_images = sorted(set(preload.train_images) | {
        row for row in preload.test_images if row[0] in targets or row[1] in unpaired_categories
    })
    return mixed
