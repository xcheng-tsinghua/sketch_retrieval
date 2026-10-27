"""Read-only, path-indexed tensors shared by DataLoader worker processes."""
import torch
from tqdm import tqdm


class SharedTensorCache:
    def __init__(self, paths, loader, *, image=False, grayscale=False, name='data'):
        paths = sorted(set(paths))
        if not paths:
            raise ValueError('Cannot cache an empty dataset')
        self.indices = {path: index for index, path in enumerate(paths)}
        self.image = image
        self.grayscale = grayscale
        self.storage = None
        for index, path in enumerate(tqdm(paths, desc=f'Caching {name}')):
            value = loader(path).detach().cpu()
            if image:
                if grayscale:
                    value = value[:1]
                value = value.mul(255).round().to(torch.uint8)
            else:
                value = value.to(torch.float32)
            if self.storage is None:
                shape = (len(paths), *value.shape)
                size_gib = len(paths) * value.numel() * value.element_size() / 1024**3
                print(f'{name}: {len(paths)} unique files, {size_gib:.3f} GiB shared cache')
                try:
                    self.storage = torch.empty(shape, dtype=value.dtype).share_memory_()
                except RuntimeError as exc:
                    raise RuntimeError(
                        f'Cannot allocate {size_gib:.3f} GiB shared cache for {name}. '
                        'Check available RAM and shared-memory capacity (Linux /dev/shm), '
                        'or use --cache_mode none.'
                    ) from exc
            if value.shape != self.storage.shape[1:]:
                raise ValueError(f'Inconsistent cached shape for {path}: {value.shape}')
            self.storage[index].copy_(value)

    def __call__(self, path):
        value = self.storage[self.indices[path]]
        if self.image:
            value = value.to(torch.float32).div_(255)
            if self.grayscale:
                value = value.expand(3, -1, -1)
            return value
        # Callers must not be able to modify the shared cache.
        return value.clone()
