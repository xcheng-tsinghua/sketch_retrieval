import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}#, "swish": swish
NORM2FN = {'BN1d':nn.BatchNorm1d, 'BN2d':nn.BatchNorm2d, 'LN':nn.LayerNorm}


class SketchEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SketchEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward(self, input_states):
        return self.embedding(input_states)


class SketchDiscreteEmbedding(nn.Module):
    '''
    max_size[tuple](x_length, y_length)
    '''
    def __init__(self, max_size, type_size, hidden_dim, pool_type):
        super(SketchDiscreteEmbedding, self).__init__()
        self.x_embedding = nn.Embedding(2*max_size[0]+2, hidden_dim//2)
        self.y_embedding = nn.Embedding(2*max_size[1]+2, hidden_dim//2)
        self.type_embedding = nn.Embedding(type_size+1, hidden_dim)
        assert pool_type in ['sum', 'con']
        self.pool_type = pool_type

    '''
    input_states[batch, seq_len, 3(input_dim)](Inputs are encoded as discrete type)
    '''
    def forward(self, input_states):
        input_states = input_states.to(dtype=torch.long)
        input_states = input_states + 1
        x_hidden = self.x_embedding(input_states[:,:,0])
        y_hidden = self.y_embedding(input_states[:,:,1])
        axis_hidden = torch.cat([x_hidden, y_hidden], dim=2)

        type_hidden = self.type_embedding(input_states[:,:,2])

        if self.pool_type == 'sum':
            return axis_hidden + type_hidden
        elif self.pool_type == 'con':
            return torch.cat([axis_hidden, type_hidden], dim=2)


class SketchSinPositionEmbedding(nn.Module):
    def __init__(self, max_length, pos_hidden_dim):
        super(SketchSinPositionEmbedding, self).__init__()
        self.pos_embedding_matrix = torch.zeros(max_length, pos_hidden_dim)
        pos_vector = torch.arange(max_length).view(max_length, 1).type(torch.float)
        dim_vector = torch.arange(pos_hidden_dim).type(torch.float) + 1.0
        self.pos_embedding_matrix[:,::2] = torch.sin(pos_vector / (dim_vector[::2] / 2).view(1, -1))
        self.pos_embedding_matrix[:,1::2] = torch.cos(pos_vector / ((dim_vector[1::2] - 1) / 2).view(1, -1))
    '''
    Input:
        position_labels[batch, seq_len]
    Output:
        position_states[batch, seq_len, pos_hidden_dim]
    '''
    def forward(self, position_labels):
        return self.pos_embedding_matrix[position_labels.view(-1),:].view(position_labels.size(0), position_labels.size(1), -1)


class SketchLearnPositionEmbedding(nn.Module):
    def __init__(self, max_length, pos_hidden_dim):
        super(SketchLearnPositionEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_length, pos_hidden_dim)

    '''
    Input:
        position_labels[batch, seq_len]
    Output:
        position_states[batch, seq_len, pos_hidden_dim]
    '''
    def forward(self, position_labels):
        return self.pos_embedding(position_labels)


class SketchEmbeddingRefineNetwork(nn.Module):
    '''
    The module to upsample the embedding feature, idea from the ALBERT: Factorized Embedding
    '''
    def __init__(self, out_dim, layers_dim):
        super(SketchEmbeddingRefineNetwork, self).__init__()
        self.layers = []
        layers_dim = layers_dim.copy()
        layers_dim.append(out_dim)

        for i in range(len(layers_dim)-1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_state):
        x = input_state
        for layer in self.layers:
            x = layer(x)
        return x


def setting2dict(paras, setting):
    paras['num_heads'] = setting[0]
    paras['hidden_dim'] = setting[1]
    paras['inter_dim'] = setting[2]


class SketchSelfAttention(nn.Module):
    '''
    Implementation for self attention in Sketch.
    The input will be a K-Dim feature.
    Input Parameters:
        config[dict]:
            hidden_dim[int]: The dimension of input hidden embeddings in the self attention, hidden diension is equal to the output dimension
            num_heads[int]: The number of heads
            attention_probs[float]: probability parameter for dropout
    '''
    def __init__(self, num_heads, hidden_dim, attention_dropout_prob):
        super(SketchSelfAttention, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, num_heads))
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        #self.attention_dropout_prob = config.attention_dropout_prob
        # Calculation for intermeidate parameters
        self.head_dim = int(self.hidden_dim / self.num_heads)
        self.all_head_dim = self.head_dim * self.num_heads
        self.scale_factor = math.sqrt(self.head_dim)

        self.query = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.key = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.value = nn.Linear(self.hidden_dim, self.all_head_dim)
        self.dropout = nn.Dropout(attention_dropout_prob)
        self.multihead_output = None

    def transpose_(self, x):
        '''
        Transpose Function for simplicity.
        '''
        new_x_shape = x.size()[:-1] + (self.num_heads , self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False, keep_multihead_output=False):
        '''
        Input:
            hidden_states[batch, seq_len, hidden_dim]
            attention_mask[batch,  1, 1, seq_len]
        Output:
            context_states[batch, seq_len, hidden_dim]
            attention_probs[seq_len, hidden_dim]
        '''
        # Get query, key, value together
        query = self.query(hidden_states) # [batch, seq_len, all_head_dim]
        key = self.key(hidden_states) # [batch, seq_len, all_head_dim]
        value = self.value(hidden_states) # [batch, seq_len, all_head_dim]

        # tranpose the query, key, value into multi heads[batch, seq_len, ]
        multi_query = self.transpose_(query) # [batch, num_heads, seq_len, head_dim]
        multi_key = self.transpose_(key) # [batch,  num_heads, seq_len, head_dim]
        multi_value = self.transpose_(value) # [batch, num_heads, seq_len, head_dim]

        # Calculate Attention maps
        attention_scores = torch.matmul(multi_query, multi_key.transpose(-1, -2))
        attention_scores = attention_scores / self.scale_factor
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # Compute states values
        context_states = torch.matmul(attention_probs, multi_value)

        if keep_multihead_output:
            self.multihead_output = context_states
            self.multihead_output.retain_grad()

        context_states = context_states.permute(0,2,1,3)
        context_states = context_states.contiguous().view(context_states.size()[:-2]+(-1,)) #view(context_states.size()[:-2]+ (self.all_head_dim,))

        if output_attentions:
            return context_states, attention_probs
        return context_states


class SketchMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_dim,
                 attention_norm_type, attention_dropout_prob, hidden_dropout_prob,):
        super(SketchMultiHeadAttention, self).__init__()
        self.attention = SketchSelfAttention(num_heads, hidden_dim, attention_dropout_prob)
        self.output = SketchOutput(hidden_dim, hidden_dim, attention_norm_type, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False):
        input_states = hidden_states
        hidden_states = self.attention(hidden_states, attention_mask, head_mask=head_mask)
        if output_attentions:
            hidden_states, attention_probs = hidden_states

        output_states = self.output(hidden_states, input_states)
        if output_attentions:
            return output_states, attention_probs

        return output_states


class SketchIntermediate(nn.Module):
    def __init__(self, hidden_dim, inter_dim, inter_activation):
        super(SketchIntermediate, self).__init__()
        self.fc = nn.Linear(hidden_dim, inter_dim)
        self.activation = ACT2FN[inter_activation]


    def forward(self, hidden_states):

        hidden_states = hidden_states.to(next(self.fc.parameters()).device)

        inter_states = self.fc(hidden_states.contiguous())
        inter_states = self.activation(inter_states)
        return inter_states


class SketchOutput(nn.Module):
    def __init__(self, input_dim, output_dim, attention_norm_type, output_dropout_prob):
        super(SketchOutput, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if attention_norm_type not in NORM2FN:
            raise ValueError(
                "The attention normalization is not in standard normalization types.")
        self.norm = NORM2FN[attention_norm_type](output_dim)
        self.dropout = nn.Dropout(output_dropout_prob)
    '''
    Input:
        hidden_states[]:

    Output:
        hidden_states[]:
    '''
    def forward(self, hidden_states, input_states):
        hidden_states = self.fc(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.norm(hidden_states+input_states)
        return hidden_states


class SketchLayer(nn.Module):
    '''
        A transformer layer for sketch bert
    '''
    def __init__(self, num_heads, hidden_dim, inter_dim,
                 attention_norm_type, inter_activation, attention_dropout_prob,
                 hidden_dropout_prob, output_dropout_prob,):
        super(SketchLayer, self).__init__()
        self.attention = SketchMultiHeadAttention(num_heads, hidden_dim,
                                                    attention_norm_type, attention_dropout_prob, hidden_dropout_prob,)
        self.inter_layer = SketchIntermediate(hidden_dim, inter_dim, inter_activation)
        self.output = SketchOutput(inter_dim, hidden_dim, attention_norm_type, output_dropout_prob)


    '''
    Input:
        hidden_states[batch, seq_len, hidden_dim]:
        attention_mask[batch, seq_len]


    '''
    def forward(self, hidden_states, attention_mask, head_mask=None, output_attentions=False):

        hidden_states = self.attention(hidden_states, attention_mask, head_mask)
        if output_attentions:
            hidden_states, attention_probs = hidden_states

        inter_states = self.inter_layer(hidden_states)
        output_states = self.output(inter_states, hidden_states)

        if output_attentions:
            return output_states, attention_probs

        return output_states


class SketchALEncoder(nn.Module):
    '''
        A Lite BERT: Parameter Sharing
        layers_setting[list]: [[12, ], []]
    '''
    def __init__(self, layers_setting,
                     attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob,):
        super(SketchALEncoder, self).__init__()
        layer_paras = {
                      'attention_norm_type':attention_norm_type, 'inter_activation':inter_activation, 'attention_dropout_prob':attention_dropout_prob,
                     'hidden_dropout_prob':hidden_dropout_prob, 'output_dropout_prob':output_dropout_prob}
        setting2dict(layer_paras, layers_setting[0])
        self.sketch_layer = SketchLayer(**layer_paras)
        self.layers = []
        for layer_setting in layers_setting:
            self.layers.append(self.sketch_layer)
        #self.layers = nn.ModuleList(self.layers)

    def forward(self, input_states, attention_mask, head_mask=None, output_all_states=False, output_attentions=False, keep_multihead_output=False):
        all_states = []
        all_attention_probs = []
        hidden_states = input_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

            if output_all_states:
                all_states.append(hidden_states)

        if not output_all_states:
            all_states.append(hidden_states)

        if output_attentions:
            return all_states, all_attention_probs

        return all_states


class SketchEncoder(nn.Module):
    '''
        layers_setting[list]: [[12, ], []]
    '''
    def __init__(self, layers_setting,
                     attention_norm_type, inter_activation, attention_dropout_prob,
                    hidden_dropout_prob, output_dropout_prob,):
        super(SketchEncoder, self).__init__()
        layer_paras = {
                      'attention_norm_type':attention_norm_type, 'inter_activation':inter_activation, 'attention_dropout_prob':attention_dropout_prob,
                     'hidden_dropout_prob':hidden_dropout_prob, 'output_dropout_prob':output_dropout_prob}
        self.layers = []
        for layer_setting in layers_setting:
            setting2dict(layer_paras, layer_setting)
            self.layers.append(SketchLayer(**layer_paras))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input_states, attention_mask, head_mask=None, output_all_states=False, output_attentions=False, keep_multihead_output=False):
        all_states = []
        all_attention_probs = []
        hidden_states = input_states
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, head_mask=head_mask, output_attentions=output_attentions)
            if output_attentions:
                hidden_states, attention_probs = hidden_states
                all_attention_probs.append(attention_probs)

            if output_all_states:
                all_states.append(hidden_states)

        if not output_all_states:
            all_states.append(hidden_states)

        if output_attentions:
            return all_states, all_attention_probs

        return all_states


class SketchTransformerModel(nn.Module):
    '''
    Input:
        layers_setting[list]
        input_dim[int]
        max_length[int]
        position_type[str]
        attention_norm_type[str]
        inter_activation[str]
        attention_dropout_prob[float]
        hidden_dropout_prob[float]
        output_dropout_prob[float]
    '''
    def __init__(self,
                 model_type='albert',
                 layers_setting=[[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072],[12, 768, 3072]],
                 embed_layers_setting=[128,256,512],
                 input_dim=5,
                 max_length=250 + 0 + 0,
                 max_size=[128, 128],
                 type_size=3,
                 position_type='learn',
                 segment_type='none',
                 sketch_embed_type='linear',
                 embed_pool_type='sum',
                 attention_norm_type='LN',
                 inter_activation='gelu',
                 attention_dropout_prob=0.5,
                 hidden_dropout_prob=0.5,
                 output_dropout_prob=0.5
                 ):
        super().__init__()

        self.layers_setting = layers_setting
        self.num_hidden_layers = len(layers_setting)
        self.embed_pool_type = embed_pool_type
        assert sketch_embed_type in ['linear', 'discrete']

        if sketch_embed_type == 'linear':
            self.embedding = SketchEmbedding(input_dim, embed_layers_setting[0])
        elif sketch_embed_type == 'discrete':
            self.embedding = SketchDiscreteEmbedding(max_size, type_size, embed_layers_setting[0], embed_pool_type)
        assert position_type in ['sin', 'learn', 'none']

        if position_type == 'sin':
            self.pos_embedding = SketchSinPositionEmbedding(max_length, embed_layers_setting[0])
        elif position_type == 'learn':
            self.pos_embedding = SketchLearnPositionEmbedding(max_length, embed_layers_setting[0])
        else:
            self.pos_embedding = None
        if segment_type == 'learn':
            self.segment_embedding = SketchLearnPositionEmbedding(max_length, embed_layers_setting[0])
        else:
            self.segment_embedding = None

        self.embed_refine_net = SketchEmbeddingRefineNetwork(layers_setting[0][1], embed_layers_setting)

        assert model_type in ['albert', 'bert']
        if model_type == 'albert':
            self.encoder = SketchALEncoder(layers_setting,
                            attention_norm_type, inter_activation, attention_dropout_prob,
                            hidden_dropout_prob, output_dropout_prob)
        elif model_type == 'bert':
            self.encoder = SketchEncoder(layers_setting,
                            attention_norm_type, inter_activation, attention_dropout_prob,
                            hidden_dropout_prob, output_dropout_prob)

    def load_model(self, state_dict, own_rel_in_input, own_cls_in_input, pre_rel_in_input, pre_cls_in_input):
        own_state = self.state_dict()
        for k, v in own_state.items():
            if k == 'pos_embedding.pos_embedding.weight':
                own_pos_size = v.size(0)
                seq_len = own_pos_size - own_rel_in_input - own_cls_in_input
                pretrained_pos_size = state_dict[k].size(0)
                own_start_ind = int(own_rel_in_input+own_cls_in_input)
                pre_start_ind = int(pre_rel_in_input+pre_cls_in_input)
                seq_len = min(seq_len, state_dict[k].size(0)-pre_start_ind)
                own_state[k][own_start_ind:own_start_ind+seq_len] = state_dict[k][pre_start_ind:pre_start_ind+seq_len]
                if own_rel_in_input and own_cls_in_input:
                    if pre_cls_in_input and pre_cls_in_input:
                        own_state[k][:2] = state_dict[k][:2]
                    elif pre_cls_in_input:
                        own_state[k][1] = state_dict[k][0]
                    elif pre_rel_in_input:
                        own_state[k][0] = state_dict[k][0]
                elif own_rel_in_input:
                    if pre_rel_in_input:
                        own_state[k][0] = state_dict[k][0]
                elif own_cls_in_input:
                    if pre_cls_in_input:
                        own_state[k][0] = state_dict[k][int(pre_rel_in_input)]
            else:
                own_state[k] = state_dict[k]
        self.load_state_dict(own_state)

    def get_pos_states(self, input_states):
        return torch.arange(input_states.size(1)).view(1,-1).repeat(input_states.size(0),1).to(device=input_states.device)
    '''
    Input:
        input_states[batch, seq_len, 5],
        attention_mask[batch, seq_len]/[batch, seq_len, ],(length mask)
    Output:
        output_states[batch, seq_len, hidden_dim],
    '''
    def forward(self, input_states, attention_mask, segments=None, head_mask=None,
                output_all_states=False, output_attentions=False, keep_multihead_output=False):
        if attention_mask is None:
            attention_mask = torch.ones(input_states.size(0), input_states.size(1))
        # Extending attention mask
        if len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask
        # process head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype, device=input_states.device) # switch to fload if need + fp16 compatibility
        else:
            head_mask = None

        input_states = self.embedding(input_states)

        if self.pos_embedding is not None:
            pos_states = self.pos_embedding(self.get_pos_states(input_states))
            input_states = input_states + pos_states.to(device=input_states.device)

        if self.segment_embedding is not None and segments is not None:
            segment_states = self.segment_embedding(segments)
            input_states = input_states + segment_states
        input_states = self.embed_refine_net(input_states)
        output_states = self.encoder(input_states, attention_mask, head_mask, output_all_states, output_attentions, keep_multihead_output)

        if output_attentions:
            output_states, attention_probs = output_states
            return output_states[-1], attention_probs

        return output_states[-1]


class SketchBERT(nn.Module):
    def __init__(self, max_length):
        super().__init__()

        self.sketch_transformer = SketchTransformerModel(max_length=max_length)
        self.linear = nn.Sequential(
            nn.Linear(768, 627),
            nn.BatchNorm1d(627),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(627, 512)
        )

    def forward(self, input_states, attention_mask):

        fea = self.sketch_transformer(input_states, attention_mask)

        # fea = fea.max(1)[0] 
        
        # ✅ 使用mask加权平均池化
        attention_mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        masked_fea = fea * attention_mask  # 应用mask
        sum_fea = masked_fea.sum(dim=1)  # [batch, 768]
        lengths = attention_mask.sum(dim=1)  # [batch, 1]
        fea = sum_fea / (lengths + 1e-8)  # 避免除零，得到平均特征
        
        fea = self.linear(fea)

        return fea


if __name__ == '__main__':
    atensor = torch.rand(10, 256, 5)
    amask = torch.rand(10, 256)

    anet = SketchBERT(256)

    aout = anet(atensor, amask)

    print(aout.size())


    pass




# packge
# import os
# import sys
# import copy
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
#
# class full_connected_conv2d(nn.Module):
#     def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4, final_proc=False):
#         '''
#         构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
#         :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
#         :param drop_rate: dropout 概率
#         '''
#         super().__init__()
#
#         self.linear_layers = nn.ModuleList()
#         self.batch_normals = nn.ModuleList()
#         self.activates = nn.ModuleList()
#         self.drop_outs = nn.ModuleList()
#         self.n_layers = len(channels)
#
#         self.final_proc = final_proc
#         if drop_rate == 0:
#             self.is_drop = False
#         else:
#             self.is_drop = True
#
#         for i in range(self.n_layers - 2):
#             self.linear_layers.append(nn.Conv2d(channels[i], channels[i + 1], 1, bias=bias))
#             self.batch_normals.append(nn.BatchNorm2d(channels[i + 1]))
#             self.activates.append(nn.LeakyReLU(negative_slope=0.2))
#             self.drop_outs.append(nn.Dropout2d(drop_rate))
#
#         self.outlayer = nn.Conv2d(channels[-2], channels[-1], 1, bias=bias)
#
#         self.outbn = nn.BatchNorm2d(channels[-1])
#         self.outat = nn.LeakyReLU(negative_slope=0.2)
#         self.outdp = nn.Dropout2d(drop_rate)
#
#     def forward(self, embeddings):
#         '''
#         :param embeddings: [bs, fea_in, n_row, n_col]
#         :return: [bs, fea_out, n_row, n_col]
#         '''
#         fea = embeddings
#         for i in range(self.n_layers - 2):
#             fc = self.linear_layers[i]
#             bn = self.batch_normals[i]
#             at = self.activates[i]
#             dp = self.drop_outs[i]
#
#             if self.is_drop:
#                 fea = dp(at(bn(fc(fea))))
#             else:
#                 fea = at(bn(fc(fea)))
#
#         fea = self.outlayer(fea)
#
#         if self.final_proc:
#             fea = self.outbn(fea)
#             fea = self.outat(fea)
#
#             if self.is_drop:
#                 fea = self.outdp(fea)
#
#         return fea
#
#
# class full_connected_conv1d(nn.Module):
#     def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4, final_proc=False):
#         '''
#         构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
#         :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
#         :param drop_rate: dropout 概率
#         '''
#         super().__init__()
#
#         self.linear_layers = nn.ModuleList()
#         self.batch_normals = nn.ModuleList()
#         self.activates = nn.ModuleList()
#         self.drop_outs = nn.ModuleList()
#         self.n_layers = len(channels)
#
#         self.final_proc = final_proc
#         if drop_rate == 0:
#             self.is_drop = False
#         else:
#             self.is_drop = True
#
#         for i in range(self.n_layers - 2):
#             self.linear_layers.append(nn.Conv1d(channels[i], channels[i + 1], 1, bias=bias))
#             self.batch_normals.append(nn.BatchNorm1d(channels[i + 1]))
#             self.activates.append(nn.LeakyReLU(negative_slope=0.2))
#             self.drop_outs.append(nn.Dropout1d(drop_rate))
#
#         self.outlayer = nn.Conv1d(channels[-2], channels[-1], 1, bias=bias)
#
#         self.outbn = nn.BatchNorm1d(channels[-1])
#         self.outat = nn.LeakyReLU(negative_slope=0.2)
#         self.outdp = nn.Dropout1d(drop_rate)
#
#     def forward(self, embeddings):
#         '''
#         :param embeddings: [bs, fea_in, n_points]
#         :return: [bs, fea_out, n_points]
#         '''
#         fea = embeddings
#         for i in range(self.n_layers - 2):
#             fc = self.linear_layers[i]
#             bn = self.batch_normals[i]
#             at = self.activates[i]
#             dp = self.drop_outs[i]
#
#             if self.is_drop:
#                 fea = dp(at(bn(fc(fea))))
#             else:
#                 fea = at(bn(fc(fea)))
#
#         fea = self.outlayer(fea)
#
#         if self.final_proc:
#             fea = self.outbn(fea)
#             fea = self.outat(fea)
#
#             if self.is_drop:
#                 fea = self.outdp(fea)
#
#         return fea
#
#
# class full_connected(nn.Module):
#     def __init__(self, channels: list, bias: bool = True, drop_rate: float = 0.4, final_proc=False):
#         '''
#         构建全连接层，输出层不接 BatchNormalization、ReLU、dropout、SoftMax、log_SoftMax
#         :param channels: 输入层到输出层的维度，[in, hid1, hid2, ..., out]
#         :param drop_rate: dropout 概率
#         '''
#         super().__init__()
#
#         self.linear_layers = nn.ModuleList()
#         self.batch_normals = nn.ModuleList()
#         self.activates = nn.ModuleList()
#         self.drop_outs = nn.ModuleList()
#         self.n_layers = len(channels)
#
#         self.final_proc = final_proc
#         if drop_rate == 0:
#             self.is_drop = False
#         else:
#             self.is_drop = True
#
#         for i in range(self.n_layers - 2):
#             self.linear_layers.append(nn.Linear(channels[i], channels[i + 1], bias=bias))
#             self.batch_normals.append(nn.BatchNorm1d(channels[i + 1]))
#             self.activates.append(nn.LeakyReLU(negative_slope=0.2))
#             self.drop_outs.append(nn.Dropout(drop_rate))
#
#         self.outlayer = nn.Linear(channels[-2], channels[-1], bias=bias)
#
#         self.outbn = nn.BatchNorm1d(channels[-1])
#         self.outat = nn.LeakyReLU(negative_slope=0.2)
#         self.outdp = nn.Dropout1d(drop_rate)
#
#     def forward(self, embeddings):
#         '''
#         :param embeddings: [bs, fea_in, n_points]
#         :return: [bs, fea_out, n_points]
#         '''
#         fea = embeddings
#         for i in range(self.n_layers - 2):
#             fc = self.linear_layers[i]
#             bn = self.batch_normals[i]
#             at = self.activates[i]
#             dp = self.drop_outs[i]
#
#             if self.is_drop:
#                 fea = dp(at(bn(fc(fea))))
#             else:
#                 fea = at(bn(fc(fea)))
#
#         fea = self.outlayer(fea)
#
#         if self.final_proc:
#             fea = self.outbn(fea)
#             fea = self.outat(fea)
#
#             if self.is_drop:
#                 fea = self.outdp(fea)
#
#         return fea
#
#
#
# # def get_graph_feature(x, k=20):
# #     """
# #     输入点云，利用knn计算每个点的邻近点，然后计算内个点中心点到邻近点的向量，再将该向量与中心点坐标拼接
# #     :param x: [bs, channel, npoint]
# #     :param k:
# #     :return: [bs, channel+channel, npoint, k]
# #     """
# #     bs, channel, npoints = x.size()
# #
# #     # -> [bs, npoint, 3]
# #     x = x.permute(0, 2, 1)
# #
# #     # -> [bs, npoint, k]
# #     idx = utils.knn(x, k)
# #
# #     # -> [bs, npoint, k, 3]
# #     point_neighbors = utils.index_points(x, idx)
# #
# #     # -> [bs, npoint, k, 3]
# #     x = x.view(bs, npoints, 1, channel).repeat(1, 1, k, 1)
# #
# #     # 计算从中心点到邻近点的向量，再与中心点拼接起来
# #     # -> [bs, npoint, k, 3]
# #     feature = torch.cat((point_neighbors - x, x), dim=3)
# #
# #     # -> [bs, 3, npoint, k]
# #     feature = feature.permute(0, 3, 1, 2)
# #     return feature
#
#
# def knn(x, k):
#     # -> x: [bs, 2, n_point]
#
#     inner = -2 * torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
#
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx
#
#
# def get_graph_feature(x, k=20, idx=None):
#     # -> x: [bs, 2, n_point]
#
#     batch_size, channel, num_points = x.size()
#
#     x = x.view(batch_size, -1, num_points)
#     if idx is None:
#         idx = knn(x, k=k)  # (batch_size, num_points, k)
#     device = x.device
#
#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
#
#     idx = idx + idx_base
#
#     idx = idx.view(-1)
#
#     _, num_dims, _ = x.size()
#
#     x = x.transpose(2, 1).contiguous()
#     feature = x.view(batch_size * num_points, -1)[idx, :]
#     feature = feature.view(batch_size, num_points, k, num_dims)
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
#
#     feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
#
#     return feature
#
#
# class DGCNN(nn.Module):
#     def __init__(self, output_channels, n_near=10, emb_dims=256, dropout=0.5):
#         super(DGCNN, self).__init__()
#         # self.k = n_near
#         #
#         # self.conv1 = utils.full_connected_conv2d([4, 8, 16], final_proc=True, drop_rate=0)
#         # self.conv2 = utils.full_connected_conv2d([16*2, 64, 128], final_proc=True, drop_rate=0)
#         #
#         # self.conv3 = utils.full_connected_conv1d([128 + 16, (128 + 16 + emb_dims) // 2, emb_dims], final_proc=True, drop_rate=0)
#
#         self.encoder = DgcnnEncoder(2, emb_dims)
#
#         self.linear = full_connected([emb_dims, (emb_dims + output_channels) // 2, output_channels], final_proc=False, drop_rate=0)
#
#     def forward(self, x):
#         # -> x: [bs, 2, n_point]
#         # assert x.size(1) == 2
#         #
#         # # -> [bs, emb, n_point, n_neighbor]
#         # x = get_graph_feature(x, k=self.k)
#         # x = self.conv1(x)
#         #
#         # # -> [bs, emb, n_point]
#         # x1 = x.max(dim=-1, keepdim=False)[0]
#         #
#         # x = get_graph_feature(x1, k=self.k)
#         # x = self.conv2(x)
#         # x2 = x.max(dim=-1, keepdim=False)[0]
#         #
#         # # -> [bs, emb, n_point]
#         # x = torch.cat((x1, x2), dim=1)
#         #
#         # # -> [bs, emb, n_points]
#         # x = self.conv3(x)
#
#         x = self.encoder(x)
#
#         # -> [bs, emb]
#         x = torch.max(x, dim=2)[0]
#
#         x = self.linear(x)
#         x = F.log_softmax(x, dim=1)
#         return x
#
#
#         # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
#         # x = torch.cat((x1, x2), 1)
#         #
#         # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
#         # x = self.dp1(x)
#         # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
#         # x = self.dp2(x)
#         # x = self.linear3(x)
#         #
#         # x = F.log_softmax(x, dim=1)
#         # return x
#
#
# class SketchEncoder(nn.Module):
#     def __init__(self, emb_in=2, emb_out=512, n_near=10):
#         super().__init__()
#         self.n_near = n_near
#
#         emb_inc = (emb_out / (4*emb_in)) ** 0.25
#         emb_l1_0 = emb_in * 2
#         emb_l1_1 = int(emb_l1_0 * emb_inc)
#         emb_l1_2 = int(emb_l1_0 * emb_inc ** 2)
#
#         emb_l2_0 = emb_l1_2 * 2
#         emb_l2_1 = int(emb_l2_0 * emb_inc)
#         emb_l2_2 = emb_out
#
#         emb_l3_0 = emb_l2_2 + emb_l1_2
#         emb_l3_1 = int(((emb_out / emb_l3_0) ** 0.5) * emb_l3_0)
#         emb_l3_2 = emb_out
#
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#
#         self.conv1 = full_connected_conv2d([emb_l1_0, emb_l1_1, emb_l1_2],
#                                                  final_proc=True,
#                                                  drop_rate=0.0
#                                                  )
#         self.conv2 = full_connected_conv2d([emb_l2_0, emb_l2_1, emb_l2_2],
#                                                  final_proc=True,
#                                                  drop_rate=0.0
#                                                  )
#
#         self.conv3 = full_connected_conv1d([emb_l3_0, emb_l3_1, emb_l3_2],
#                                                  final_proc=True, drop_rate=0.0
#                                                  )
#
#     def forward(self, x):
#         # x: [bs, n_token, 2]
#
#         x = x.permute(0, 2, 1)
#         # x: [bs, 2, n_token]
#
#         # -> [bs, emb, n_token, n_neighbor]
#         x = get_graph_feature(x, k=self.n_near)
#         x = self.conv1(x)
#
#         # -> [bs, emb, n_token]
#         x1 = x.max(dim=-1, keepdim=False)[0]
#
#         x = get_graph_feature(x1, k=self.n_near)
#         x = self.conv2(x)
#         x2 = x.max(dim=-1, keepdim=False)[0]
#
#         # -> [bs, emb, n_token]
#         x = torch.cat((x1, x2), dim=1)
#
#         # -> [bs, emb, n_token]
#         x = self.conv3(x)
#
#         x = x.max(2)[0]
#
#         return x, self.logit_scale.exp()
#
#
# def test():
#
#
#     # atensor = torch.ones((2, 2, 2))
#     # btensor = torch.rand((2, 1, 1))
#     #
#     # print(torch.arange(0, 5))
#     # print(btensor)
#     # print(atensor + btensor)
#     # exit()
#
#     # btensor = torch.rand((4, 3))
#     # print(btensor)
#     # print(btensor.max(dim=-1)[0])
#     # exit()
#
#     btensor = torch.rand((2, 256, 100))
#     modelaaa = DgcnnEncoder()
#     print(modelaaa(btensor).size())
#
#     exit()
#
#     def parameter_number(__model):
#         return sum(p.numel() for p in __model.parameters() if p.requires_grad)
#
#     sys.path.append("../..")
#     import time
#
#     device = torch.device('cuda:0')
#     points = torch.randn(8, 1024, 3).to(device)
#     model = DGCNN().to(device)
#
#     start = time.time()
#     out = model(points)
#
#     print("Inference time: {}".format(time.time() - start))
#     print("Parameter #: {}".format(parameter_number(model)))
#     print("Input size: {}".format(points.size()))
#     print("Out   size: {}".format(out.size()))
#
#
# if __name__ == "__main__":
#     test()
#
