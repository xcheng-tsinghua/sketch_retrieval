import numpy as np
from torch.utils.data import Dataset
import os
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch
from functools import lru_cache
import ftfy
import regex as re
import html
import gzip
import pickle


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def __call__(self, texts, context_length=77):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            tokens = tokens[:context_length]
            result[i, :len(tokens)] = torch.tensor(tokens)

        if len(result) == 1:
            return result[0]
        return result


def get_subdirs(dir_path):
    """
    获取 dir_path 的所有一级子文件夹
    仅仅是文件夹名，不是完整路径
    """
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    '''
    获取dir_path下的全部文件路径
    '''
    filepath_all = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.split('.')[-1] == suffix:
                if filename_only:
                    current_filepath = file
                else:
                    current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    return filepath_all

class SketchImageDataset(Dataset):
    """
    专门用于草图-图片跨模态对齐的数据集类
    适用于预处理后的sketchy数据集
    
    数据集结构：
    - 草图路径: E:\Master\Experiment\data\stroke-normal
      └─ category_name/
          └─ sketch_file.npy (stroke-5格式，已归一化)
    - 图片路径: E:\Master\Experiment\data\photo  
      └─ category_name/
          └─ image_file.jpg
    """
    
    def __init__(self, 
                 n_skh_points=256,
                 is_train=True,
                 fixed_split_path='E:\\Master\\Frontier\\AIGC\\Omni\\sketch_large_model\\data\\fixed_splits\\sketch_image_dataset_splits.pkl'):
        """
        Args:
            n_skh_points: 草图点数
            is_train: 是否为训练集
            fixed_split_path: 固定数据划分文件路径
        """
        
        print(f'SketchImageDataset initialized with:')
        print(f'  Mode: {"train" if is_train else "test"}')
        print(f'  Fixed split path: {fixed_split_path}')
        
        self.n_skh_points = n_skh_points
        self.is_train = is_train
        
        # 使用预先保存的固定数据划分
        if os.path.exists(fixed_split_path):
            print(f'Loading fixed dataset split from: {fixed_split_path}')
            self._load_fixed_split(fixed_split_path)
        else:
            raise FileNotFoundError(f'Fixed split file not found at {fixed_split_path}. Please create the fixed split file first.')
        
        print(f'Dataset size: {len(self.data_pairs)} pairs')
        
        # 图片预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
    
    def _load_fixed_split(self, fixed_split_path):
        """加载预先保存的固定数据划分"""
        import pickle
        
        with open(fixed_split_path, 'rb') as f:
            dataset_info = pickle.load(f)
        
        # 根据is_train选择对应的数据
        if self.is_train:
            self.data_pairs = dataset_info['train_pairs']
        else:
            self.data_pairs = dataset_info['test_pairs']
        
        print(f'Loaded fixed split: {len(self.data_pairs)} pairs')
        print(f'Total categories: {dataset_info["total_categories"]}')
    
    def load_sketch_stroke5(self, sketch_path):
        """
        加载stroke-5格式的草图数据
        stroke-5格式: [x, y, p1, p2, p3] 其中p1,p2,p3是pen state
        使用完全确定性的采样策略
        """
        try:
            sketch_data = np.load(sketch_path)  # 加载.npy文件
            
            # 如果点数超过限制，进行随机性采样
            # if len(sketch_data) > self.n_skh_points:
            #     choice = np.random.choice(len(sketch_data), self.n_skh_points, replace=True)   
            #     sketch_data = sketch_data[choice, :]

            # 创建固定大小的tensor
            sketch_tensor = torch.zeros(self.n_skh_points, 5, dtype=torch.float32)
            mask = torch.zeros(self.n_skh_points, dtype=torch.float32)
            
            actual_len = min(len(sketch_data), self.n_skh_points)
            sketch_tensor[:actual_len] = torch.from_numpy(sketch_data[:actual_len]).float()
            mask[:actual_len] = 1.0
            
            return sketch_tensor, mask
            
        except Exception as e:
            print(f"Error loading sketch {sketch_path}: {e}")
            # 返回空的tensor
            sketch_tensor = torch.zeros(self.n_skh_points, 5, dtype=torch.float32)
            mask = torch.zeros(self.n_skh_points, dtype=torch.float32)
            return sketch_tensor, mask
    
    def __getitem__(self, index):
        """
        返回: (sketch_data, sketch_mask, image_data, category)
        """
        sketch_path, image_path, category = self.data_pairs[index]
        
        # 加载草图数据
        sketch_data, sketch_mask = self.load_sketch_stroke5(sketch_path)
        
        # 加载图片数据
        try:
            image = Image.open(image_path).convert("RGB")
            image_data = self.image_transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回黑色图片
            image_data = torch.zeros(3, 224, 224, dtype=torch.float32)
        
        return sketch_data, sketch_mask, image_data, category
    
    def __len__(self):
        return len(self.data_pairs)


if __name__ == '__main__':
    # 测试SketchImageDataset
    try:
        dataset = SketchImageDataset(is_train=True)
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        if len(dataset) > 0:
            sketch, mask, image, category = dataset[0]
            print(f"Sample 0:")
            print(f"  Sketch shape: {sketch.shape}")
            print(f"  Mask shape: {mask.shape}")  
            print(f"  Image shape: {image.shape}")
            print(f"  Category: {category}")
    except Exception as e:
        print(f"Error testing SketchImageDataset: {e}")
    
    # 原有的测试代码
    # adataset = SLMataset()
    # adata = adataset[0]
    # print('text emb: ', adata[0].size())
    # print('point cloud: ', adata[1].shape)
    # print('sketch points: ', adata[2].shape)
    # print('image: ', adata[3].size())

    pass









