"""Exact retrieval metrics, with query chunking to bound similarity memory."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_retrieval(sketch_features, image_features, task, paired_indices,
                       sketch_labels, image_labels, topk=(1, 5, 10),
                       chunk_size=128, map_k=200):
    if task not in ('cl', 'fg') or chunk_size < 1:
        raise ValueError('Invalid task or chunk size')
    queries = F.normalize(sketch_features.cpu(), dim=1)
    gallery = F.normalize(image_features.cpu(), dim=1)
    nq, ng = len(queries), len(gallery)
    if not nq or not ng:
        raise ValueError('Evaluation requires non-empty queries and gallery')
    targets = torch.as_tensor(paired_indices, dtype=torch.long)
    qlabels = torch.as_tensor(sketch_labels, dtype=torch.long)
    glabels = torch.as_tensor(image_labels, dtype=torch.long)
    if len(targets) != nq or len(qlabels) != nq or len(glabels) != ng:
        raise ValueError('Feature/label lengths disagree')
    if task == 'fg' and ((targets < 0).any() or (targets >= ng).any()):
        raise ValueError('FG query target is absent from the gallery')
    successes = {k: [] for k in topk}
    sums = {'mAP': 0., f'mAP@{map_k}': 0., f'P@{map_k}': 0.}
    cutoff = min(map_k, ng)
    ranks = torch.arange(1, ng + 1).float()
    for start in range(0, nq, chunk_size):
        stop = min(start + chunk_size, nq)
        scores = queries[start:stop] @ gallery.T
        order = scores.argsort(dim=1, descending=True, stable=True)
        if task == 'cl':
            relevant = glabels[order] == qlabels[start:stop, None]
        else:
            relevant = order == targets[start:stop, None]
        counts = relevant.sum(dim=1)
        if (counts == 0).any():
            raise ValueError('Query has no relevant gallery image')
        for k in topk:
            hit = relevant[:, :min(k, ng)].any(dim=1)
            successes[k].append(hit.nonzero(as_tuple=True)[0] + start)
        precision = relevant.cumsum(dim=1) / ranks
        contributions = precision * relevant
        sums['mAP'] += (contributions.sum(dim=1) / counts).sum().item()
        # AP@K denominator: min(total relevant images, K), not retrieved positives.
        sums[f'mAP@{map_k}'] += (contributions[:, :cutoff].sum(dim=1) /
                                 counts.clamp(max=cutoff)).sum().item()
        sums[f'P@{map_k}'] += relevant[:, :cutoff].float().mean(dim=1).sum().item()
    successes = {k: torch.cat(v) for k, v in successes.items()}
    metrics = {f'Acc@{k}': len(v) / nq for k, v in successes.items()}
    metrics.update({name: value / nq for name, value in sums.items()})
    return metrics, successes
