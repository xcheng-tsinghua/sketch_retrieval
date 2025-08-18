import os
import numpy as np
from torch.utils import data

from data import sketchy_configs as scfg


def load_data_test(args):
    pre_load = PreLoad(args)
    sk_valid_data = ValidSet(pre_load, 'sk', half=True)
    im_valid_data = ValidSet(pre_load, 'im', half=True)
    return sk_valid_data, im_valid_data


def load_data(args):
    pre_load = PreLoad(args)
    train_data = TrainSet(args, scfg.sketchy_train_classes, pre_load)
    sk_valid_data = ValidSet(pre_load, 'sk')
    im_valid_data = ValidSet(pre_load, 'im')
    return train_data, sk_valid_data, im_valid_data


def get_file_iccv(labels, rootpath, class_name, cname, number, file_ls):
    """
    从该类别随机选取一个样本，并转化为对应的完整文件路径
    """
    # 该类的label
    label = np.argwhere(cname == class_name)[0, 0]
    # 该类的所有样本
    ind = np.argwhere(labels == label)
    ind_rand = np.random.randint(1, len(ind), number)
    ind_ori = ind[ind_rand]
    files = file_ls[ind_ori][0][0]
    full_path = os.path.join(rootpath, files)
    return full_path


def get_file_list_iccv(args, rootpath, skim, split):

    if args.dataset == 'sketchy_extend':
        if args.test_class == "test_class_sketchy25":
            shot_dir = "zeroshot1"
        elif args.test_class == "test_class_sketchy21":
            shot_dir = "zeroshot0"
        else:
            NameError("zeroshot is invalid")

        if skim == 'sketch':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/all_photo_filelist_zero.txt'

    elif args.dataset == 'tu_berlin':
        if skim == 'sketch':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/png_ready_filelist_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/ImageResized_ready_filelist_zero.txt'

    elif args.dataset == 'Quickdraw':
        if skim == 'sketch':
            file_ls_file = args.data_path + f'/QuickDraw/zeroshot/sketch_zero.txt'
        elif skim == 'images':
            file_ls_file = args.data_path + f'/QuickDraw/zeroshot/all_photo_zero.txt'

    else:
        NameError(args.dataset + 'is invalid')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()
    file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])
    file_names = np.array([(rootpath + x) for x in file_ls])

    # 对验证的样本数量进行缩减
    # sketch 15229->762 image 17101->1711
    if args.dataset == 'sketchy_extend' and split == 'test' and skim == 'sketch':
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 15229
        else:
            index = [i for i in range(0, file_names.shape[0], 20)]   # 762
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == 'sketchy_extend' and split == 'test' and skim == 'images':
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 17101
        else:
            index = [i for i in range(0, file_names.shape[0], 10)]  # 1711
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    # sketch 2400->800, image 27989->1400
    if args.dataset == "tu_berlin" and skim == "sketch" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 2400
        else:
            index = [i for i in range(0, file_names.shape[0], 3)]  # 800
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == "tu_berlin" and skim == "images" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 27989
        else:
            index = [i for i in range(0, file_names.shape[0], 20)]  # 1400
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    # Quickdraw 92291->770, image 54151->1806
    if args.dataset == "Quickdraw" and skim == "sketch" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 92291
        else:
            index = [i for i in range(0, file_names.shape[0], 120)]  # 770
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    if args.dataset == "Quickdraw" and skim == "images" and split == "test":
        if args.testall:
            index = [i for i in range(0, file_names.shape[0], 1)]  # 54151
        else:
            index = [i for i in range(0, file_names.shape[0], 30)]  # 1806
        file_names = file_names[index[:]]
        labels = labels[index[:]]

    file_names_cls = labels
    return file_names, file_names_cls


def get_all_train_file(args, skim):
    if skim != 'sketch' or skim != 'image':
        NameError(skim + ' not implemented!')

    if args.dataset == 'sketchy_extend':
        if args.test_class == "test_class_sketchy25":
            shot_dir = "zeroshot1"
        elif args.test_class == "test_class_sketchy21":
            shot_dir = "zeroshot0"

        cname_cid = args.data_path + f'/Sketchy/{shot_dir}/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/sketch_tx_000000000000_ready_filelist_train.txt'
        elif skim == 'image':
            file_ls_file = args.data_path + f'/Sketchy/{shot_dir}/all_photo_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif args.dataset == 'tu_berlin':
        cname_cid = args.data_path + '/TUBerlin/zeroshot/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/png_ready_filelist_train.txt'
        elif skim == 'image':
            file_ls_file = args.data_path + '/TUBerlin/zeroshot/ImageResized_ready_filelist_train.txt'
        else:
            NameError(skim + ' not implemented!')

    elif args.dataset == 'Quickdraw':
        cname_cid = args.data_path + '/QuickDraw/zeroshot/cname_cid.txt'
        if skim == 'sketch':
            file_ls_file = args.data_path + '/QuickDraw/zeroshot/sketch_train.txt'
        elif skim == 'image':
            file_ls_file = args.data_path + '/QuickDraw/zeroshot/all_photo_train.txt'
        else:
            NameError(skim + ' not implemented!')

    else:
        NameError(skim + ' not implemented! ')

    with open(file_ls_file, 'r') as fh:
        file_content = fh.readlines()

    # 图片相对路径
    file_ls = np.array([' '.join(ff.strip().split()[:-1]) for ff in file_content])
    # 图片的label,0,1,2...
    labels = np.array([int(ff.strip().split()[-1]) for ff in file_content])

    # 所有的训练类
    with open(cname_cid, 'r') as ci:
        class_and_indx = ci.readlines()
    # 类名
    cname = np.array([' '.join(cc.strip().split()[:-1]) for cc in class_and_indx])

    return file_ls, labels, cname


def create_dict_texts(texts):
    texts = list(texts)
    dicts = {l: i for i, l in enumerate(texts)}
    return dicts


class PreLoad:
    def __init__(self, args):
        self.all_valid_or_test_sketch = []
        self.all_valid_or_test_sketch_label = []
        self.all_valid_or_test_image = []
        self.all_valid_or_test_image_label = []

        self.all_train_sketch = []
        self.all_train_sketch_label = []
        self.all_train_image = []
        self.all_train_image_label = []

        self.all_train_sketch_cls_name = []
        self.all_train_image_cls_name = []

        self.init_valid_or_test(args)
        # load_para(args)

    def init_valid_or_test(self, args):
        if args.dataset == 'sketchy_extend':
            train_dir = args.data_path + '/Sketchy/'
        elif args.dataset == 'tu_berlin':
            train_dir = args.data_path + '/TUBerlin/'
        elif args.dataset == 'Quickdraw':
            train_dir = args.data_path + '/QuickDraw/'
        else:
            NameError("Dataset is not implemented")

        self.all_valid_or_test_sketch, self.all_valid_or_test_sketch_label = \
            get_file_list_iccv(args, train_dir, "sketch", "test")
        self.all_valid_or_test_image, self.all_valid_or_test_image_label = \
            get_file_list_iccv(args, train_dir, "images", "test")

        self.all_train_sketch, self.all_train_sketch_label, self.all_train_sketch_cls_name =\
            get_all_train_file(args, "sketch")
        self.all_train_image, self.all_train_image_label, self.all_train_image_cls_name = \
            get_all_train_file(args, "image")

        print("used for valid or test sketch / image:")
        print(self.all_valid_or_test_sketch.shape, self.all_valid_or_test_image.shape)
        print("used for train sketch / image:")
        print(self.all_train_sketch.shape, self.all_train_image.shape)


class TrainSet(data.Dataset):
    def __init__(self, root, train_class_label, pre_load):
        self.root = root
        self.pre_load = pre_load
        self.train_class_label = train_class_label
        self.choose_label = []
        self.class_dict = create_dict_texts(train_class_label)

    def __getitem__(self, index):
        # 随机选取三个类别
        self.choose_label_name = np.random.choice(self.train_class_label, 3, replace=False)

        sk_label = self.class_dict.get(self.choose_label_name[0])
        im_label = self.class_dict.get(self.choose_label_name[0])
        sk_label_neg = self.class_dict.get(self.choose_label_name[0])
        im_label_neg = self.class_dict.get(self.choose_label_name[-1])

        sketch = get_file_iccv(self.pre_load.all_train_sketch_label, self.root, self.choose_label_name[0],
                               self.pre_load.all_train_sketch_cls_name, 1, self.pre_load.all_train_sketch)

        image = get_file_iccv(self.pre_load.all_train_image_label, self.root, self.choose_label_name[0],
                              self.pre_load.all_train_image_cls_name, 1, self.pre_load.all_train_image)

        sketch_neg = get_file_iccv(self.pre_load.all_train_sketch_label, self.root, self.choose_label_name[0],
                                   self.pre_load.all_train_sketch_cls_name, 1, self.pre_load.all_train_sketch)

        image_neg = get_file_iccv(self.pre_load.all_train_image_label, self.root, self.choose_label_name[-1],
                                  self.pre_load.all_train_image_cls_name, 1, self.pre_load.all_train_image)

        sketch = preprocess(sketch, 'sk')
        image = preprocess(image)
        sketch_neg = preprocess(sketch_neg, 'sk')
        image_neg = preprocess(image_neg)

        return sketch, image, sketch_neg, image_neg, \
               sk_label, im_label, sk_label_neg, im_label_neg

    def __len__(self):
        return self.args.datasetLen


class ValidSet(data.Dataset):

    def __init__(self, pre_load, type_skim='im', half=False, path=False):
        self.type_skim = type_skim
        self.half = half
        self.path = path
        if type_skim == "sk":
            self.file_names, self.cls = pre_load.all_valid_or_test_sketch, pre_load.all_valid_or_test_sketch_label
        elif type_skim == "im":
            self.file_names, self.cls = pre_load.all_valid_or_test_image, pre_load.all_valid_or_test_image_label
        else:
            NameError(type_skim + " is not right")

    def __getitem__(self, index):
        label = self.cls[index]  # label 为数字
        file_name = self.file_names[index]
        if self.path:
            image = file_name
        else:
            if self.half:
                image = preprocess(file_name, self.type_skim).half()
            else:
                image = preprocess(file_name, self.type_skim)
        return image, label

    def __len__(self):
        return len(self.file_names)

