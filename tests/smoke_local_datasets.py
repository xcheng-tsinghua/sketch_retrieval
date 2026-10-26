"""Real-data GPU smoke test; reads source datasets without modifying them.

Run from repository root: python -m tests.smoke_local_datasets
Uses random initialization by default: this validates wiring, not retrieval quality.
"""
import argparse
import copy
import gc
import json
import tempfile
from pathlib import Path

import torch

import options
from data import retrieval_datasets
from encoders.sbir_model_wrapper import create_sbir_model_wrapper
from utils.trainer import SBIRTrainer


def small_split(preload):
    subset = copy.copy(preload)
    for split in ('train', 'test'):
        records = getattr(preload, split + '_pairs')
        gallery = getattr(preload, split + '_images')
        categories = sorted({r[2] for r in records})[:2]
        selected = []
        for cat in categories:
            seen = set()
            for row in records:
                if row[2] == cat and row[1] not in seen:
                    selected.append(row)
                    seen.add(row[1])
                    if len(seen) == (2 if len(categories) > 1 else 4):
                        break
        paths = {r[1] for r in selected}
        images = [r for r in gallery if r[0] in paths]
        images += [r for r in gallery if r[0] not in paths and r[1] in categories][:2]
        setattr(subset, split + '_pairs', selected)
        setattr(subset, split + '_images', images)
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketch-model', default='vit')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--output', default='log/local_smoke_results.json')
    cfg = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for name in ('sketchy_cl_zs', 'sketchy_fg_zs', 'qmul_shoe_fg', 'qmul_chair_fg'):
        args = options.parse_args(['--is_load_config', 'True', '--configs_name', name + '.yaml',
                                   '--sketch_model', cfg.sketch_model])
        preload = retrieval_datasets.preload_from_args(args)
        train_categories = {r[2] for r in preload.train_pairs}
        test_categories = {r[2] for r in preload.test_pairs}
        if eval(args.is_zero_shot):
            assert not train_categories & test_categories
        else:
            train_ids = {(Path(r[1]).name, r[2]) for r in preload.train_pairs}
            test_ids = {(Path(r[1]).name, r[2]) for r in preload.test_pairs}
            assert not train_ids & test_ids
        counts = {split: {'queries': len(getattr(preload, split + '_pairs')),
                          'images': len(getattr(preload, split + '_images'))}
                  for split in ('train', 'test')}
        subset = small_split(preload)
        info = options.get_encoder_info(args.sketch_model)
        train_loader, test_loader = retrieval_datasets.create_sketch_image_dataloaders(
            2, 0, subset, info['sketch_format'], False, args.task, n_pos=2, n_neg=2)
        model = create_sbir_model_wrapper(32, args.sketch_model, 'vit',
                                          sketch_format=info['sketch_format'], pretrained=cfg.pretrained)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = SBIRTrainer(model, train_loader, test_loader, device,
                                  str(Path(tmp) / 'smoke.pth'), None, name, 1e-4, 1e-4, 1,
                                  args.task, eval_chunk_size=2)
            before = model.image_projection[0].weight.detach().clone()
            loss = trainer.train_epoch()
            assert torch.isfinite(torch.tensor(loss))
            assert not torch.equal(before, model.image_projection[0].weight)
            trainer.validate_epoch()
            results[name] = {'full_split_counts': counts, 'smoke_train_loss': loss,
                             'smoke_metrics': trainer.last_metrics, 'device': str(device),
                             'sketch_model': cfg.sketch_model, 'pretrained': cfg.pretrained}
            print(name, json.dumps(results[name]), flush=True)
        del trainer, model, train_loader, test_loader, preload, subset, before
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
