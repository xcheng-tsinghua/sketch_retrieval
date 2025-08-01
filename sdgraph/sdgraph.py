import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

import global_defs
import sdgraph.sdgraph_utils as su
import sdgraph.utils as eu


class DownSample(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_in,  # 输入通道数 (RGB)
                out_channels=dim_out,  # 输出通道数 (保持通道数不变)
                kernel_size=(1, 3),  # 卷积核大小：1x3，仅在宽度方向上滑动
                stride=(1, 2),  # 步幅：高度方向为1，宽度方向为2
                padding=(0, 1)  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dim_out),
            eu.activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk, n_stk_pnt // 2]
        """
        dense_fea = self.conv(dense_fea)
        return dense_fea


class UpSample(nn.Module):
    """
    对sdgraph同时进行上采样
    上采样后笔划数及笔划上的点数均变为原来的2倍
    """
    def __init__(self, dim_in, dim_out, dropout=0.4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=dim_in,  # 输入通道数
                out_channels=dim_out,  # 输出通道数
                kernel_size=(1, 4),  # 卷积核大小：1x2，仅在宽度方向扩展
                stride=(1, 2),  # 步幅：高度不变，宽度扩展为原来的 2 倍
                padding=(0, 1),  # 填充：在宽度方向保持有效中心对齐
            ),
            nn.BatchNorm2d(dim_out),
            eu.activate_func(),
            nn.Dropout2d(dropout)
        )

    def forward(self, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :return: [bs, emb, n_stk, n_stk_pnt * 2]
        """
        dense_fea = self.conv(dense_fea)
        return dense_fea


class SparseToDense(nn.Module):
    """
    将sgraph转移到dgraph
    直接拼接到该笔划对应的点
    """
    def __init__(self):
        super().__init__()

    def forward(self, sparse_fea, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk, n_stk_pnt]
        """
        dense_feas_from_sparse = einops.repeat(sparse_fea, 'b c s -> b c s sp', sp=dense_fea.size(3))

        # -> [bs, emb, n_stk, n_stk_pnt]
        union_dense = torch.cat([dense_fea, dense_feas_from_sparse], dim=1)

        return union_dense


class DenseToSparse(nn.Module):
    """
    将 dense graph 的数据转移到 sparse graph
    通过卷积然后最大池化到一个特征，然后拼接
    """
    def __init__(self):
        super().__init__()

        # self.n_stk = n_stk
        # self.n_stk_pnt = n_stk_pnt

        # self.dense_to_sparse = nn.Sequential(
        #     nn.Conv2d(in_channels=dense_in, out_channels=dense_in, kernel_size=(1, 3)),
        #     nn.BatchNorm2d(dense_in),
        #     activate_func(),
        #     nn.Dropout2d(dropout),
        # )

    def forward(self, sparse_fea, dense_fea):
        """
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param sparse_fea: [bs, emb, n_stk]
        :return: [bs, emb, n_stk]
        """
        # -> [bs, emb, n_stk, n_stk_pnt - 2]
        # dense_fea = self.dense_to_sparse(dense_fea)

        # -> [bs, emb, n_stk]
        sparse_feas_from_dense = dense_fea.max(3)[0]
        # assert sparse_feas_from_dense.size(2) == self.n_stk

        # -> [bs, emb, n_stk]
        union_sparse = torch.cat([sparse_fea, sparse_feas_from_dense], dim=1)

        return union_sparse


class SDGraphEncoder(nn.Module):
    def __init__(self,
                 sparse_in, sparse_out, dense_in, dense_out,  # 输入输出维度
                 # n_stk, n_stk_pnt,  # 笔划数，每个笔划中的点数
                 sp_near=2, dn_near=10,  # 更新sdgraph的两个GCN中邻近点数目
                 sample_type='down_sample',  # 采样类型
                 with_time=False, time_emb_dim=0,  # 是否附加时间步
                 dropout=0.4
                 ):
        """
        :param sample_type: [down_sample, up_sample, none]
        """
        super().__init__()
        # self.n_stk = n_stk
        # self.n_stk_pnt = n_stk_pnt
        self.with_time = with_time

        self.dense_to_sparse = DenseToSparse()  # 这个不能设为零
        self.sparse_to_dense = SparseToDense()

        self.sparse_update = su.SparseUpdate(sparse_in + dense_in, sparse_out, sp_near)
        self.dense_update = su.DenseUpdate(dense_in + sparse_in, dense_out, dn_near)

        self.sample_type = sample_type
        if self.sample_type == 'down_sample':
            self.sample = DownSample(dense_out, dense_out, dropout)  # 这里dropout不能为零
        elif self.sample_type == 'up_sample':
            self.sample = UpSample(dense_out, dense_out, dropout)  # 这里dropout不能为零
        elif self.sample_type == 'none':
            self.sample = nn.Identity()
        else:
            raise ValueError('invalid sample type')

        if self.with_time:
            self.time_mlp_sp = su.TimeMerge(sparse_out, sparse_out, time_emb_dim, dropout)  # 这里dropout不能为零
            self.time_mlp_dn = su.TimeMerge(dense_out, dense_out, time_emb_dim, dropout)  # 这里dropout不能为零

    def forward(self, sparse_fea, dense_fea, time_emb=None):
        """
        :param sparse_fea: [bs, emb, n_stk]
        :param dense_fea: [bs, emb, n_stk, n_stk_pnt]
        :param time_emb: [bs, emb]
        :return: [bs, emb, n_stk], [bs, emb, n_stk, n_stk_pnt]
        """

        # 信息交换
        union_sparse = self.dense_to_sparse(sparse_fea, dense_fea)
        union_dense = self.sparse_to_dense(sparse_fea, dense_fea)

        # 信息更新
        union_sparse = self.sparse_update(union_sparse)
        union_dense = self.dense_update(union_dense)

        # 采样
        union_dense = self.sample(union_dense)

        # 融合时间步特征
        assert self.with_time ^ (time_emb is None)
        if self.with_time:
            union_sparse = self.time_mlp_sp(union_sparse, time_emb)
            union_dense = self.time_mlp_dn(union_dense, time_emb)

        return union_sparse, union_dense


class SDGraphCls(nn.Module):
    """
    对于每个子模块的输入和输出
    sparse graph 统一使用 [bs, channel, n_stk]
    dense graph 统一使用 [bs, channel, n_stk, n_stk_pnt]
    """
    def __init__(self, n_class: int, channel_in=2, dropout: float = 0.4):
        """
        :param n_class: 总类别数
        """
        super().__init__()
        print('cls valid')

        self.n_stk = global_defs.n_stk
        self.n_stk_pnt = global_defs.n_stk_pnt

        # 各层特征维度
        sparse_l0 = 32 + 16
        sparse_l1 = 128 + 64
        sparse_l2 = 512 + 256

        dense_l0 = 32
        dense_l1 = 128
        dense_l2 = 512

        # sparse_l0 = 16 + 8
        # sparse_l1 = 64 + 32
        # sparse_l2 = 256 + 128
        #
        # dense_l0 = 16
        # dense_l1 = 64
        # dense_l2 = 256

        # 生成初始 sdgraph
        self.point_to_sparse = su.PointToSparse(channel_in, sparse_l0)
        self.point_to_dense = su.PointToDense(channel_in, dense_l0)

        # 利用 sdgraph 更新特征
        self.sd1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                  # self.n_stk, self.n_stk_pnt,
                                  dropout=dropout
                                  )

        self.sd2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                  # self.n_stk, self.n_stk_pnt // 2,
                                  dropout=dropout
                                  )

        # 利用输出特征进行分类
        sparse_glo = sparse_l0 + sparse_l1 + sparse_l2
        dense_glo = dense_l0 + dense_l1 + dense_l2
        out_inc = (n_class / (sparse_glo + dense_glo)) ** (1 / 3)

        out_l0 = sparse_glo + dense_glo
        out_l1 = int(out_l0 * out_inc)
        out_l2 = int(out_l1 * out_inc)
        out_l3 = n_class

        self.linear = eu.MLP(0, (out_l0, out_l1, out_l2, out_l3), True, dropout, False)

    def forward(self, xy, mask=None):
        """
        :param xy: [bs, n_stk, n_stk_pnt, 2]
        :param mask: 占位用
        :return: [bs, n_classes]
        """
        bs, n_stk, n_stk_pnt, channel = xy.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt and channel == 2

        # 生成初始 sparse graph
        sparse_graph0 = self.point_to_sparse(xy)
        assert sparse_graph0.size()[2] == self.n_stk

        # 生成初始 dense graph
        dense_graph0 = self.point_to_dense(xy)
        assert dense_graph0.size()[2] == n_stk and dense_graph0.size()[3] == n_stk_pnt

        # 交叉更新数据
        sparse_graph1, dense_graph1 = self.sd1(sparse_graph0, dense_graph0)
        sparse_graph2, dense_graph2 = self.sd2(sparse_graph1, dense_graph1)

        # 提取全局特征
        sparse_glo0 = sparse_graph0.max(2)[0]
        sparse_glo1 = sparse_graph1.max(2)[0]
        sparse_glo2 = sparse_graph2.max(2)[0]

        dense_glo0 = dense_graph0.amax((2, 3))
        dense_glo1 = dense_graph1.amax((2, 3))
        dense_glo2 = dense_graph2.amax((2, 3))

        all_fea = torch.cat([sparse_glo0, sparse_glo1, sparse_glo2, dense_glo0, dense_glo1, dense_glo2], dim=1)

        # 利用全局特征分类
        cls = self.linear(all_fea)
        cls = F.log_softmax(cls, dim=1)

        return cls


class SDGraphUNet(nn.Module):
    def __init__(self, channel_in, channel_out, n_stk=global_defs.n_stk, n_stk_pnt=global_defs.n_stk_pnt, dropout=0.0):
        super().__init__()
        print('diff valid')

        '''草图参数'''
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.n_stk = n_stk
        self.n_stk_pnt = n_stk_pnt

        '''各层通道数'''
        sparse_l0 = 32
        sparse_l1 = 128
        sparse_l2 = 512

        dense_l0 = 16
        dense_l1 = 64
        dense_l2 = 256

        global_dim = 1024
        time_emb_dim = 256

        '''时间步特征生成层'''
        self.time_encode = su.TimeEncode(time_emb_dim)

        '''点坐标 -> 初始 sdgraph 生成层'''
        self.point_to_sparse = su.PointToSparse(channel_in, sparse_l0, with_time=True, time_emb_dim=time_emb_dim)
        self.point_to_dense = su.PointToDense(channel_in, dense_l0, with_time=True, time_emb_dim=time_emb_dim)

        '''下采样层 × 2'''
        self.sd_down1 = SDGraphEncoder(sparse_l0, sparse_l1, dense_l0, dense_l1,
                                       sp_near=1, dn_near=50,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        self.sd_down2 = SDGraphEncoder(sparse_l1, sparse_l2, dense_l1, dense_l2,
                                       sp_near=1, dn_near=50,
                                       sample_type='down_sample',
                                       with_time=True, time_emb_dim=time_emb_dim,
                                       dropout=dropout)

        '''全局特征生成层'''
        global_in = sparse_l2 + dense_l2
        self.global_linear = eu.MLP(dimension=0,
                                    channels=(global_in, int((global_in * global_dim) ** 0.5), global_dim),
                                    final_proc=True,
                                    dropout=dropout
                                    )

        '''上采样层 × 2'''
        self.sd_up2 = SDGraphEncoder(global_dim + sparse_l2, sparse_l2,
                                     global_dim + dense_l2, dense_l2,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        self.sd_up1 = SDGraphEncoder(sparse_l1 + sparse_l2, sparse_l1,
                                     dense_l1 + dense_l2, dense_l1,
                                     sp_near=2, dn_near=10,
                                     sample_type='up_sample',
                                     with_time=True, time_emb_dim=time_emb_dim,
                                     dropout=dropout)

        '''最终输出层'''
        final_in = dense_l0 + sparse_l0 + dense_l1 + sparse_l1 + channel_in
        self.final_linear = eu.MLP(dimension=1,
                                   channels=(final_in, int((channel_out * final_in) ** 0.5), channel_out),
                                   final_proc=False,
                                   dropout=dropout)

    def img_size(self):
        return self.n_stk, self.n_stk_pnt, self.channel_out

    def forward(self, xy, time):
        """
        :param xy: [bs, n_stk, n_stk_pnt, channel_in]
        :param time: [bs, ]
        :return: [bs, n_stk, n_stk_pnt, channel_out]
        """
        '''生成时间步特征'''
        time_emb = self.time_encode(time)

        bs, n_stk, n_stk_pnt, channel_in = xy.size()
        assert n_stk == self.n_stk and n_stk_pnt == self.n_stk_pnt and channel_in == self.channel_in

        '''生成初始 sdgraph'''
        sparse_graph_up0 = self.point_to_sparse(xy, time_emb)  # -> [bs, emb, n_stk]
        dense_graph_up0 = self.point_to_dense(xy, time_emb)  # -> [bs, emb, n_point]
        assert sparse_graph_up0.size()[2] == self.n_stk and dense_graph_up0.size()[2] == n_stk and dense_graph_up0.size()[3] == n_stk_pnt

        '''下采样'''
        sparse_graph_up1, dense_graph_up1 = self.sd_down1(sparse_graph_up0, dense_graph_up0, time_emb)
        sparse_graph_up2, dense_graph_up2 = self.sd_down2(sparse_graph_up1, dense_graph_up1, time_emb)

        '''获取全局特征'''
        sp_up2_glo = sparse_graph_up2.max(2)[0]
        dn_up2_glo = dense_graph_up2.amax((2, 3))

        fea_global = torch.cat([sp_up2_glo, dn_up2_glo], dim=1)
        fea_global = self.global_linear(fea_global)  # -> [bs, emb]

        '''将 sd_graph 融合全局特征 (直接拼接在后面)'''
        sparse_fit = einops.repeat(fea_global, 'b c -> b c s', s=self.n_stk)
        sparse_graph_down2 = torch.cat([sparse_graph_up2, sparse_fit], dim=1)

        dense_fit = einops.repeat(fea_global, 'b c -> b c s sp', s=dense_graph_up2.size(2), sp=dense_graph_up2.size(3))
        dense_graph_down2 = torch.cat([dense_graph_up2, dense_fit], dim=1)

        '''上采样并融合UNet下采样阶段对应特征'''
        sparse_graph_down1, dense_graph_down1 = self.sd_up2(sparse_graph_down2, dense_graph_down2, time_emb)  # -> [bs, sp_l2, n_stk], [bs, dn_l2, n_pnt]

        sparse_graph_down1 = torch.cat([sparse_graph_down1, sparse_graph_up1], dim=1)
        dense_graph_down1 = torch.cat([dense_graph_down1, dense_graph_up1], dim=1)

        sparse_graph_down0, dense_graph_down0 = self.sd_up1(sparse_graph_down1, dense_graph_down1, time_emb)

        sparse_graph = torch.cat([sparse_graph_down0, sparse_graph_up0], dim=1)
        dense_graph = torch.cat([dense_graph_down0, dense_graph_up0], dim=1)

        '''将sparse graph及xy转移到dense graph并输出'''
        sparse_graph = einops.repeat(sparse_graph, 'b c s -> b c s sp', sp=self.n_stk_pnt)
        xy = xy.permute(0, 3, 1, 2)  # -> [bs, channel, n_stk, n_stk_pnt]

        dense_graph = torch.cat([dense_graph, sparse_graph, xy], dim=1)
        dense_graph = dense_graph.view(bs, dense_graph.size(1), self.n_stk * self.n_stk_pnt)

        noise = self.final_linear(dense_graph)  # -> [bs, channel_out, n_stk * n_stk_pnt]
        noise = einops.rearrange(noise, 'b c (s sp) -> b s sp c', s=self.n_stk, sp=self.n_stk_pnt)

        return noise


def test():
#     bs = 3
#     atensor = torch.rand([bs, 2, global_defs.n_skh_pnt]).cuda()
#     t1 = torch.randint(0, 1000, (bs,)).long().cuda()
#
#     # classifier = SDGraphSeg2(2, 2).cuda()
#     # cls11 = classifier(atensor, t1)
#
#     classifier = SDGraphCls2(10).cuda()
#     cls11 = classifier(atensor)
#
#     print(cls11.size())
#
#     print('---------------')


    bs = 3
    atensor = torch.rand([bs, global_defs.n_stk, global_defs.n_stk_pnt, 2])
    t1 = torch.randint(0, 1000, (bs,)).long()

    classifier_unet = SDGraphUNet(2, 2)
    cls11 = classifier_unet(atensor, t1)
    print(cls11.size())

    classifier_cls = SDGraphCls(10)
    cls12 = classifier_cls(atensor)
    print(cls12.size())

    print('---------------')








if __name__ == '__main__':
    test()
    print('---------------')





