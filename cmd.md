cd /opt/data/private/networks/sketch_retrieval/ && conda activate dp

git pull && git status && git add . && git commit -m "change" && git push

pkill -9 -f "python train.py"

## 查看 nohup 的进程输出
tail -f nohup.out

## 查看 nohup 的进程
ps -ef | grep python

nohup 末尾添加该命令可以指定log文件
> out.log 2>&1 &
例如：
nohup python train.py > out.log 2>&1 &



# 对于目录
# 需要将 ${directory} 更换为已被Git同步但是需要解除同步的文件夹
git rm --cached -r ${directory}

# 对于文件
# 需要将 ${file} 更换为已被Git同步但是需要解除同步的文件
git rm --cached ${file}


## 创建新分支

1. 查看当前分支

git branch

2. 基于当前分支创建新分支

git branch v2026-07-26

3. 切换到新分支

git switch v2026-07-26

4. 将本地分支关联到远程分支，并推送新分支到远程仓库

git push -u origin v2026-07-26

5. 备份完成后切换回主分支

git switch main

## 将当前分支强制覆盖 main 分支

1. 假设用于覆盖 main 的分支叫 edge_tag
git fetch origin
git switch edge_tag

2. 可选：给旧 main 建一个备份分支
git branch backup-main origin/main
git push origin backup-main

3. 用 edge_tag 强制覆盖远程 main
git push origin edge_tag:main --force-with-lease

4. 切回 main，并同步覆盖后的内容
git switch main
git fetch origin
git reset --hard origin/main

## 将远程 main 分支备份

1. 更新远程分支信息
git fetch origin

2. 基于远程 main 创建本地备份分支
git branch valid_2026_10_26 origin/main

3. 将备份分支推送到 GitHub，并强制将本地分支关联到远程同名分支
git push -u origin valid_2026_10_26

# 训练除sdgraph外其他的模型

python train.py --sketch_model=bidir_lstm --is_load_ckpt=False && python train.py --sketch_model=vit --is_load_ckpt=False

python train.py --sketch_model=vit --root_sever=/opt/data/private/data_set/sketch_retrieval/qmul_v2_fit/shoe --add_str=shoe --is_load_ckpt=False

## TMM R1
随机种子： 92,48,69,83,5

## CL-ZS-SBIR sketchy
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=92 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=48 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=69 && \

nohup bash -c '
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=83 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=5 && \

python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --test_mix_ratio=0.3 --seed=92 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --test_mix_ratio=0.3 --seed=48 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --test_mix_ratio=0.3 --seed=69 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --test_mix_ratio=0.3 --seed=83 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --test_mix_ratio=0.3 --seed=5 && \

python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --test_mix_ratio=0.3 --seed=92 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --test_mix_ratio=0.3 --seed=48 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --test_mix_ratio=0.3 --seed=69 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --test_mix_ratio=0.3 --seed=83 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --test_mix_ratio=0.3 --seed=5
' > out.log 2>&1 &
tail -f out.log



nohup bash -c '
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=92 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=48 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=69 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=83 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=5 && \

python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=92 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=48 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=69 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=83 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=5
' > out.log 2>&1 &
tail -f out.log




nohup bash -c '
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=92 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=48 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=69 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=83 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=5 && \

python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=92 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=48 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=69 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=83 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=5 && \

python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=92 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=48 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=69 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=83 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=5
' > out_fg_zs.log 2>&1 &
tail -f out_fg_zs.log




// 2026.10.30

nohup bash -c '
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=69 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=83 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=5 && \

python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=92 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=48 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=69 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=83 && \
python train.py --task=cl --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=5
' > out.log 2>&1 &
tail -f out.log





nohup bash -c '
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=48 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=69 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=83 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=sdgraph --seed=5 && \

python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=92 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=48 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=69 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=83 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gat --seed=5 && \

python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=92 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=48 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=69 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=83 && \
python train.py --task=fg --is_zero_shot=True --root_sever=/opt/data/private/data_set/sketch_retrieval/sketchy --sketch_model=gcn --seed=5
' > out_fg_zs.log 2>&1 &
tail -f out_fg_zs.log









