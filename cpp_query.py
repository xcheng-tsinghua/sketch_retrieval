"""
用于接受C++的查询需求，返回对应的草图
"""
import numpy as np
from flask import Flask, request, jsonify
import torch
from colorama import Fore, Back, Style
from datetime import datetime
import os
from tqdm import tqdm

from data import retrieval_datasets
from encoders import sbir_model_wrapper
import options
from utils import trainer, utils


app = Flask(__name__)
args = options.parse_args()
save_str = utils.get_save_str(args)
print(Fore.BLACK + Back.CYAN + '-> model save name: ' + save_str + ' <-' + Style.RESET_ALL)


def prepare_model_and_data():
    encoder_info = options.get_encoder_info(args.sketch_model)

    # 设置日志
    os.makedirs('log', exist_ok=True)
    logger = utils.get_log('./log/' + save_str + f'-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.txt')

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预加载数据集
    root = args.root_local if eval(args.local) else args.root_sever
    pre_load = retrieval_datasets.DatasetPreload(
        sketch_root=os.path.join(root, encoder_info['sketch_subdir']),
        image_root=os.path.join(root, encoder_info['image_subdir']),
        sketch_suffix=encoder_info['sketch_suffix'],
        image_suffix=encoder_info['image_suffix'],
        is_multi_pair=True if args.pair_mode == 'multi_pair' else False,
        split_mode=args.task,
        multi_sketch_split=args.multi_sketch_split
    )

    # 创建数据加载器
    train_loader, test_loader = retrieval_datasets.create_sketch_image_dataloaders(
        batch_size=args.bs,
        num_workers=args.num_workers,
        pre_load=pre_load,
        sketch_format=encoder_info['sketch_format'],
        is_full_train=eval(args.is_full_train)
    )

    # 创建模型
    model = sbir_model_wrapper.create_sbir_model_wrapper(
        embed_dim=args.embed_dim,
        freeze_image_encoder=eval(args.is_freeze_image_encoder),
        freeze_sketch_backbone=eval(args.is_freeze_sketch_backbone),
        sketch_model_name=args.sketch_model,
        image_model_name=args.image_model,
        sketch_format=encoder_info['sketch_format'],
    )

    # 创建训练器
    check_point = utils.get_check_point(args.weight_dir, save_str)
    model_trainer = trainer.SBIRTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        check_point=check_point,
        logger=logger,
        save_str=save_str,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epoch,
    )

    if not model_trainer.load_checkpoint(check_point, eval(args.is_load_ckpt)):
        print('no valid ckpt')
        exit(0)

    # 先对训练集中的全部样本进行编码
    target_loader = model_trainer.train_loader
    model_trainer.model.eval()

    # 提取图片特征
    target_loader.dataset.back_image()
    image_features = []

    for data_tensor in tqdm(target_loader, desc='loading data'):
        # 由于 shuffle=False，无需担心加载过程中图片顺序被打乱
        data_tensor = data_tensor.to(device)

        fea_tensor = model_trainer.model.encode_image(data_tensor)
        image_features.append(fea_tensor.cpu())

    # 合并特征
    image_features = torch.cat(image_features, dim=0)

    model_trainer.model.eval()

    # 图片路径表
    image_path_list = target_loader.dataset.image_list

    return model_trainer.model, image_features, image_path_list


revl_model = None
image_features = None
image_path_list = None


# 检索样本数
n_retrieval = 5

def inference(pnt_seq):
    """
    pnt_seq: [n, 2]
    目前仅支持 vit - vit
    """
    # 先将序列转换为图片
    skh_pixel = utils.s3_to_tensor_img(pnt_seq)

    # 提取草图特征
    sketch_features = revl_model.encode_sketch(skh_pixel.unsqueeze(0))

    # 计算相似度
    sim_matrix = sketch_features @ image_features.t()  # [1, n]

    _, topk_indices = sim_matrix.topk(k=n_retrieval, dim=1, largest=True, sorted=True)  # [1, max_k]
    topk_indices = topk_indices.tolist()

    # 找到对应的字符串
    searched_img_path_list = []
    for idx in topk_indices:
        searched_img_path_list.append(image_path_list[idx])

    return searched_img_path_list


@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    x = torch.tensor(data['input'], dtype=torch.float32)
    y = inference(x)

    return jsonify({"output": y})


def test_input():
    x = np.loadtxt(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketch_cad\sketch_s3_352\Bearing\00b11be6f26c85ca85f84daf52626b36_1.txt')
    y = inference(x)

    print(y)


if __name__ == "__main__":
    revl_model, image_features, image_path_list = prepare_model_and_data()

    # app.run(host='0.0.0.0', port=5000)
    test_input()



