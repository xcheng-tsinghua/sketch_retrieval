sketch retrieval methods

'SBIR': 使用所有类别，每个类别内取出一定数量用作测试。
'ZS-SBIR': 一部分类别全部用于训练，另一部分类别全部用于测试，即训练类别和测试类别不重合

## 训练与评估

在项目根目录运行，使用本地 `dp` 环境：

```powershell
conda activate dp
python train.py --is_load_config True --configs_name sketchy_cl_zs.yaml --is_load_ckpt False
python train.py --is_load_config True --configs_name sketchy_fg_zs.yaml --is_load_ckpt False
python train.py --is_load_config True --configs_name qmul_shoe_fg.yaml --is_load_ckpt False
python train.py --is_load_config True --configs_name qmul_chair_fg.yaml --is_load_ckpt False
```

以上是四个独立实验，选择需要的命令运行。配置默认使用 ViT、预训练权重和本地数据路径；Windows 的 `num_workers` 默认设为 0。
命令行显式参数优先于 YAML，例如增加 `--sketch_model bidir_lstm --bs 8 --epoch 100`。
服务器使用 `--local False --root_sever /your/dataset/path`。
`--is_load_ckpt False` 从头训练；去掉该参数可恢复对应实验最近保存的权重、epoch、optimizer 和 scheduler。

| 配置 | 任务 | 训练草图 / 图库 | 测试草图 / 图库 |
| --- | --- | --- | --- |
| `sketchy_cl_zs.yaml` | CL-ZS-SBIR | 62,787 / 10,400 | 12,694 / 2,100 |
| `sketchy_fg_zs.yaml` | FG-ZS-SBIR | 62,787 / 10,400 | 12,694 / 2,100 |
| `qmul_shoe_fg.yaml` | FG-SBIR | 5,982 / 1,800 | 666 / 200 |
| `qmul_chair_fg.yaml` | FG-SBIR | 952 / 300 | 323 / 100 |

数量来自当前本地目录扫描，并非代码硬编码。

### 数据协议

- Sketchy 使用 `D:/document/DataSet/sketch_retrieval/sketchy` 下的 `sketch_s3_352` 和 `photo`。
  按用户选择，**仅使用此目录的 12,500 张照片**，不自动合并相邻扩展图库。
  测试类别沿用 `data/sketchy_configs.py` 中的 21 类（训练 104 类），并非自动选择其他论文的划分。
  可用 `--zs_classes_file path/to/classes.txt` 指定其他划分，每行一个目录类别名。
- QMUL 使用 `qmul_v2_fit/shoe` 或 `qmul_v2_fit/chair` 中预先划分的 `train/test` 草图。
  本地两个照片子目录都包含完整照片集合，因此默认仅保留各自草图对应的照片：防止测试图片成为训练负样本。
- `--gallery_mode auto`：FG-SBIR 使用配对图库；CL 或 zero-shot 使用对应类别/划分下的全部图库。
  `all` 保留无配对草图的图片，`paired` 只保留查询对应图片。
  如以后需要扩展图库，可显式指定 `--extended_image_root`，其目录应为 `root/category/image.jpg`。
  FG 中额外图片作为干扰项，CL 中同类额外图片也是相关结果。扩展目录不会用于解析 FG 的原始实例目标。
- 非 zero-shot 自动划分按照片实例分组，确保同一照片的多张草图不跨训练/测试集。
  zero-shot 模式检查训练/测试类别不重叠，禁止 `is_full_train=True`。
- `--sketch_subdir` 可覆盖编码器的默认草图目录。例如本地 Sketchy 的笔画数据目录是 `sketch_stk11_stkpnt32`，
  与默认 SDGraph 的 12 笔画参数不同；选择这类编码器时，目录和 `options.py` 中的笔画规格须匹配。

### CL/FG 训练与指标

CL 采样同类正图片和异类负图片，支持 `--n_pos`、`--n_neg`；多正样本逐个与负样本池计算 InfoNCE 后取平均，同类正样本之间不互斥。
FG 使用唯一配对实例作为正样本，其他图库实例作为负样本，忽略 `n_pos`。
候选样本少于指定数量时有放回采样；没有合法负样本时明确报错。

评估独立编码草图查询和去重后的图片路径图库，按查询分块计算余弦相似度。
CL 以类别相同为相关，FG 以配对图片索引相同为相关。输出 `Acc@1/5/10`、`mAP`、`mAP@200`、`P@200`：

- Acc@K 表示前 K 中至少一个相关结果的查询比例。
- AP 为所有相关位置的 Precision 平均值；AP@200 分母为 `min(图库中相关图片总数, 200)`。
- 图库不足 K 张时使用实际图库大小。FG 的单目标 mAP 等于平均倒数排名。

CL 按 mAP、FG 按 Acc@1 保存 `_best.pth`，每轮都检查是否改善。
训练仍按已有流程使用 test split 评估和选模；若需要独立无偏的最终测试成绩，应另设 validation split。

仅评估已有 checkpoint（不训练）：

```powershell
python train.py --is_load_config True --configs_name sketchy_cl_zs.yaml --eval_only --checkpoint model_trained/your_checkpoint_best.pth
```

编码器和 `embed_dim` 等结构参数必须与 checkpoint 一致。`--eval_only` 不再加载预训练初始权重。
指标输出到终端及对应 `log/` 文件。

### 验证

```powershell
python -m unittest discover -s tests -v
python -m tests.smoke_local_datasets --pretrained
```

单元测试覆盖类别/实例相关性、多正样本损失梯度、小图库指标、数据划分、防止 QMUL 跨划分图库混用、配置覆盖和 checkpoint 恢复。
真实数据 smoke test 扫描四个完整划分，再对每个任务抽取小样本，运行 ViT 前向、反向、参数更新及评估；不修改原数据。
结果保存到 `log/local_smoke_results.json`。不加 `--pretrained` 使用随机初始化，适合离线检查。
这些小样本指标仅验证流程，不是完整训练后的检索性能。

