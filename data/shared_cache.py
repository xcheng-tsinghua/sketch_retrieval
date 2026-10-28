"""Read-only, path-indexed tensors shared by DataLoader worker processes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch
from tqdm import tqdm


class SharedTensorCache:
    def __init__(self, paths, loader, *, image=False, grayscale=False, name='data',
                 cache_dir='', signature=None, rebuild=False):
        paths = sorted(set(paths))
        if not paths:
            raise ValueError('Cannot cache an empty dataset')
        self.indices = {path: index for index, path in enumerate(paths)}
        self.image = image
        self.grayscale = grayscale
        self.storage = None
        cache_path = None
        if cache_dir:
            # Metadata validation still touches the source filesystem, but avoids
            # reading/decoding the contents of every source on a warm start.
            manifest = {'version': 1, 'signature': signature, 'image': image,
                        'grayscale': grayscale, 'files': []}
            for path in paths:
                stat = os.stat(path)
                manifest['files'].append([os.path.abspath(path), stat.st_size, stat.st_mtime_ns])
            key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
            directory = Path(cache_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            cache_path = directory / f'{name}-{key}.pt'
            if cache_path.is_file() and not rebuild:
                print(f'Loading persistent {name} cache: {cache_path}', flush=True)
                try:
                    saved = torch.load(cache_path, map_location='cpu', weights_only=True)
                    tensor = saved['storage']
                    expected_dtype = torch.uint8 if image else torch.float32
                    if (saved['manifest'] != manifest or tensor.shape[0] != len(paths)
                            or tensor.dtype != expected_dtype):
                        raise ValueError('Cache metadata or tensor mismatch')
                except (OSError, RuntimeError, ValueError, KeyError, EOFError) as exc:
                    raise RuntimeError(f'Cannot read cache {cache_path}; retry with '
                                       '--cache_rebuild to replace it') from exc
                self.storage = tensor.share_memory_()
                return
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
        if cache_path is not None:
            # Publish only complete files; interrupted writes never become cache hits.
            handle, temporary = tempfile.mkstemp(prefix=cache_path.name + '.', suffix='.tmp',
                                                  dir=cache_path.parent)
            os.close(handle)
            try:
                torch.save({'manifest': manifest, 'storage': self.storage}, temporary)
                os.replace(temporary, cache_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            print(f'Saved persistent {name} cache: {cache_path}', flush=True)

    def __call__(self, path):
        value = self.storage[self.indices[path]]
        if self.image:
            value = value.to(torch.float32).div_(255)
            if self.grayscale:
                value = value.expand(3, -1, -1)
            return value
        # Callers must not be able to modify the shared cache.
        return value.clone()
