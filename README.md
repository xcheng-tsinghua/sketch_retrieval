sketch retrieval methods

'SBIR': 使用所有类别，每个类别内取出一定数量用作测试。
'ZS-SBIR': 一部分类别全部用于训练，另一部分类别全部用于测试，即训练类别和测试类别不重合

## 训练与评估

训练默认记录 W&B 日志。先安装 `pip install -r requirements-wandb.txt`，并在项目根目录
`.env` 中设置 `WANDB_API_KEY=你的密钥`。密钥仅用于登录，不写入实验配置。
project 默认是 `{FG或CL}_{ZS_SBIR或SBIR}_{数据集}`，例如
`FG_ZS_SBIR_sketchy`、`FG_SBIR_qmul_chair`、`CL_SBIR_qmul_shoe`；
根据 `--local` 选择的根目录识别数据集（不区分大小写，也支持 `qmul_v2_fit/chair` 和 `qmul_v2_fit/shoe`）。
可用 `--wandb_project` 覆盖。run name 使用 `save_str`，末尾追加 `_{seed}`；
`--seed` 默认 42，同时控制训练与数据划分。旧权重可通过 `--checkpoint` 显式指定。
每个 epoch 记录训练 loss、学习率、全部验证指标与最佳分数；训练退出时关闭 W&B run。
`--eval_only` 保持仅本地评估，不创建 W&B run。

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

### 固定训练规模的测试集混入与早停

`--test_mix_ratio` 指定新训练集内测试草图样本的比例（0 到 1，默认 0）。
原训练集有 N 个样本时，抽取 `floor(N * test_mix_ratio)` 个测试样本替换等量训练样本，
总长度仍为 N；启动时按 `--seed` 抽样一次，各 epoch 使用同一份混合集。
测试样本充足时无放回抽样，不足时有放回抽样。配对照片随测试草图加入训练图库；
无配对目标的 CL 草图加入对应类别的测试图库。原训练图库保留，图库大小不受 N 限制。
完整测试查询和测试图库保持不变。混入大于 0 时，训练会接触测试数据（包括 zero-shot 测试类别）。
`train.py` 要求 `--is_full_train False`，用此比例替代旧的全部合并选项。

`--model` 是 `--sketch_model` 的别名。例如，在现有配置上混入 10% 测试样本：

```powershell
python train.py --is_load_config True --configs_name sketchy_cl_zs.yaml --model gat --test_mix_ratio 0.1 --is_load_ckpt False
```

每个 epoch 在完整测试集评估，指标**严格大于**下表阈值时保存当前检查点并立即结束训练，不再开始下一个 epoch。
根目录按 `--local` 选择，不区分大小写；Sketchy 要求路径含 `sketchy` 且 `--is_zero_shot True`，
QMUL 要求路径同时含 `qmul` 和 `chair`/`shoe`，且 `--task fg --is_zero_shot False`。
其他任务或模型不启用阈值早停。

| 模型 | Sketchy CL-ZS mAP@200 | Sketchy FG-ZS Acc@1 | QMUL-Chair FG Acc@1 | QMUL-Shoe FG Acc@1 |
| --- | --- | --- | --- | --- |
| sdgraph | 0.763 | 0.328 | 0.724 | 0.455 |
| gat | 0.508 | 0.108 | 0.251 | 0.190 |
| gcn | 0.493 | 0.174 | 0.254 | 0.179 |
| bidir_gru | 0.455 | 0.162 | 0.238 | 0.176 |
| gru | 0.409 | 0.141 | 0.235 | 0.164 |
| bidir_lstm | 0.471 | 0.156 | 0.232 | 0.158 |
| lstm | 0.393 | 0.135 | 0.220 | 0.153 |

早停时的模型保存在普通检查点路径；`_best.pth` 继续按原规则选择（CL 为完整 mAP，FG 为 Acc@1）。
改变混入比例开展新实验时，可用 `--is_load_ckpt False --add_str mix10` 区分实验权重。

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

### GAT 草图编码器

`--sketch_model gat` 使用 [Veličković et al., Graph Attention Networks, ICLR 2018](https://arxiv.org/abs/1710.10903)
中的原始 GAT 加性注意力（式 3–6），不是 GATv2，也不是现有 `sdgraph_attn` 的点云 Transformer。
实现位于 `encoders/gat.py`，仅依赖 PyTorch，无需安装 PyG 或 DGL。

```powershell
python train.py --is_load_config True --configs_name sketchy_cl_zs.yaml --sketch_model gat --is_load_ckpt False
```

将配置名替换为 `sketchy_fg_zs.yaml`、`qmul_shoe_fg.yaml` 或 `qmul_chair_fg.yaml` 即可用于其他三种任务。
GAT 读取已有 `sketch_s3_352`，通过统一加载器得到相对坐标 S5，不需要额外预处理文件。

- 节点是有效草图点；输入特征是中心化、缩放后的绝对二维坐标及 pen-up/down 状态。
- 边包含自环、同笔画连续点双向边、对称化的空间 8 近邻边。空间边允许跨笔画连接。
- 两层 GAT：第一层 4 头，每头 32 维，拼接后 ELU；第二层 4 头，每头 64 维，取头均值。
- 有效节点均值池化后投影至 `embed_dim`，输出 L2 归一化；padding 不参与邻域或池化。
- 构图和图级池化是本项目的草图检索适配，不代表复现论文的节点分类实验或其原始超参数。
  GAT 自身从随机初始化学习；`pretrained=True` 仍控制图像 ViT 的预训练权重。
  `is_freeze_sketch_backbone=True` 冻结两层 GAT，保留 readout 可训练。

邻居数、头数和隐藏维度配置在 `options.py` 的 `supported_encoders['gat']` 中。
采用密集掩码注意力，显存随点数平方增长；默认最多 352 点。
公式一致性、掩码、笔画边界和 padding 回归测试在 `tests/test_gat.py`。
真实数据验证命令：

```powershell
python -m tests.smoke_local_datasets --sketch-model gat --pretrained --output log/local_smoke_gat_results.json
```

### GLCN（用户引用的 GCN）草图编码器

Jiang、Zhang、Lin、Tang、Luo 的 CVPR 2019 论文
[Semi-Supervised Learning With Graph Learning-Convolutional Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.html)
提出的是 **GLCN**，而不是 Kipf–Welling 的固定图 GCN。
本项目使用 `--sketch_model glcn`，同时接受 `--sketch_model gcn` 作为该实现的别名。
两个名称产生不同的 checkpoint 文件名前缀，恢复训练时应继续使用原来的名称。

```powershell
python train.py --is_load_config True --configs_name sketchy_cl_zs.yaml --sketch_model glcn --is_load_ckpt False
```

替换为其他三个配置即可训练 FG-ZS-SBIR 或 QMUL FG-SBIR。实现见 `encoders/glcn.py`，无需额外依赖。
GAT 和 GLCN 共用草图点特征及初始邻接构造函数 `build_sketch_graph`，GLCN 再学习边权：

- 将 4 维节点特征线性投影至 2 维，以 `ReLU(a^T |Px_i-Px_j|)` 计算边分数，掩码行 softmax 得到 S。
- 使用同一个 S 进行两层 `S X W` 图卷积（64、128 维，ReLU），有效点均值池化后映射到检索嵌入。
- 训练 loss 为现有 CL/FG InfoNCE 加图学习正则项：距离平方加权和与 `gamma * ||S||_F^2`。
  为适配不同长度的草图，正则项按每图有效点数归一化，再对 batch 取均值。
  默认外部权重 `graph_loss_weight=0.01`、`gamma=1.0`，可在 `options.py` 对应模型规格中修改。
- 图学习投影、边权参数和卷积参数联合优化；评估不使用图正则损失，也不使用测试标签构图。
  `is_freeze_sketch_backbone=True` 冻结图学习和卷积模块，仅保留 readout 可训练。

公式依据论文[作者预印本](https://arxiv.org/pdf/1811.09971)的式 9–13、15。
这是面向草图检索的适配：节点为草图点，训练监督为检索损失，不是原论文跨样本半监督节点分类实验的逐项复现。
GLCN 自身无预训练权重；`pretrained` 仍控制图片 ViT。密集图计算的内存随点数平方增长。

```powershell
python -m tests.smoke_local_datasets --sketch-model glcn --pretrained --output log/local_smoke_glcn_results.json
```

