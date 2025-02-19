# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 9:49 上午
# @Author  : Chongming GAO
# @FileName: state_tracker.py
import math

import torch
from timm.models.layers import trunc_normal_

from core.inputs import SparseFeatP
from deepctr_torch.inputs import varlen_embedding_lookup, get_varlen_pooling_list, \
    VarLenSparseFeat, DenseFeat, combined_dnn_input

from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from core.layers import PositionalEncoding
from core.user_model import create_embedding_matrix, build_input_features, compute_input_dim

FLOAT = torch.FloatTensor


def input_from_feature_columns(X, feature_columns, embedding_dict, feature_index, device, support_dense=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError(
            "DenseFeat is not supported in dnn_feature_columns")

    sparse_embedding_list = [embedding_dict[feat.embedding_name](
        X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
        feat in sparse_feature_columns]

    sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
                                                  varlen_sparse_feature_columns)
    varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
                                                           varlen_sparse_feature_columns, device)

    dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                        dense_feature_columns]

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list


class StateTrackerBase(nn.Module):
    def __init__(self, user_columns, action_columns, feedback_columns,
                 has_user_embedding=True, has_action_embedding=True, has_feedback_embedding=False,
                 dataset="VirtualTB-v0",
                 device='cpu', seed=2021,
                 init_std=0.0001):
        super(StateTrackerBase, self).__init__()
        torch.manual_seed(seed)

        self.dataset = dataset

        # self.user_index = build_input_features(user_columns)
        # self.action_index = build_input_features(action_columns)

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device

        self.has_user_embedding = has_user_embedding
        self.has_action_embedding = has_action_embedding
        self.has_feedback_embedding = has_feedback_embedding

        self.user_columns = user_columns
        self.action_columns = action_columns
        self.feedback_columns = feedback_columns

        all_columns = []
        if not has_user_embedding:
            all_columns += user_columns
            self.user_index = build_input_features(user_columns)
        if not has_action_embedding:
            all_columns += action_columns
            self.action_index = build_input_features(action_columns)
        if not has_feedback_embedding:
            all_columns += feedback_columns
            self.feedback_index = build_input_features(feedback_columns)

        self.embedding_dict = create_embedding_matrix(all_columns, init_std, sparse=False, device=device)

    def get_embedding(self, X, type):
        if type == "user":
            has_embedding = self.has_user_embedding
        elif type == "action":
            has_embedding = self.has_action_embedding
        elif type == "feedback":
            has_embedding = self.has_feedback_embedding
        if has_embedding:
            return FLOAT(X).to(self.device)

        if type == "user":
            feat_columns = self.user_columns
            feat_index = self.user_index
        elif type == "action":
            feat_columns = self.action_columns
            feat_index = self.action_index
        elif type == "feedback":
            feat_columns = self.feedback_columns
            feat_index = self.feedback_index

        sparse_embedding_list, dense_value_list = \
            input_from_feature_columns(FLOAT(X).to(self.device), feat_columns, self.embedding_dict, feat_index,
                                       self.device)

        new_X = combined_dnn_input(sparse_embedding_list, dense_value_list)

        return new_X

    def build_state(self,
                    obs=None,
                    env_id=None,
                    obs_next=None,
                    rew=None,
                    done=None,
                    info=None,
                    policy=None):
        return {}


class StateTrackerTransformer(StateTrackerBase):
    def __init__(self, user_columns, action_columns, feedback_columns,
                 dim_model, dim_state, dim_max_batch, dropout=0.1,
                 dataset="VirtualTB-v0",
                 has_user_embedding=True, has_action_embedding=True, has_feedback_embedding=False,
                 nhead=8, d_hid=128, nlayers=2,
                 device='cpu', seed=2021,
                 init_std=0.0001, padding_idx=None, MAX_TURN=100):

        super(StateTrackerTransformer, self).__init__(user_columns, action_columns, feedback_columns,
                                                      has_user_embedding=has_user_embedding,
                                                      has_action_embedding=has_action_embedding,
                                                      has_feedback_embedding=has_feedback_embedding,
                                                      dataset=dataset,
                                                      device=device, seed=seed, init_std=init_std)
        self.dim_model = dim_model
        self.MAX_TURN = MAX_TURN + 1  # For user will take an additional slut.

        self.ffn_user = nn.Linear(compute_input_dim(user_columns),
                                  dim_model, device=device)
        # self.fnn_gate = nn.Linear(3 * compute_input_dim(action_columns),
        #                           dim_model, device=device)
        self.fnn_gate = nn.Linear(1 + compute_input_dim(action_columns),
                                  dim_model, device=device)
        self.gate = nn.Linear( compute_input_dim(action_columns) + dim_model,
                                  dim_model, device=device)
        self.sigmoid = nn.Sigmoid()

        self.pos_encoder = PositionalEncoding(dim_model, dropout, max_len=self.MAX_TURN)
        encoder_layers = TransformerEncoderLayer(dim_model, nhead, d_hid, dropout)
        #layers = EncoderLayer(dim_model, d_hid, dropout)
        self.encoder_layer = Encode_layer(dim=dim_model,drop = dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = Encoder(dim_model,dropout,nlayers)
        self.gru = StateGRU(dim_model, d_hid)
        self.asb = Adaptive_Spectral_Block(dim_model)
        self.norm1 = nn.LayerNorm(dim_model)

        self.fc1 = nn.Linear(dim_model, dim_model)
        self.fc2 = nn.Linear(dim_model, dim_model)

        self.decoder = nn.Linear(dim_model, dim_state)
        self.dim_state = dim_state

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1

        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src0: Tensor,  src_mask:Tensor) -> Tensor: # src_mask:Tensor
        """
        Args:
            src0: Tensor, shape [seq_len, batch_size, dim]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.dim_model)
        src = src0 * math.sqrt(self.dim_model)  # Added by Chongming
        src_p = self.pos_encoder(src)
        #output = self.transformer_encoder(src_p, src_mask)  #[seq_len, batch_size, dim]
        output = self.gru(src_p)
        output_t = output[-1, :, :]

        s_t = self.decoder(output_t)
        return s_t

    def build_state(self, obs=None,  #[batch,1]  分批次的id值，只用于初始化
                    env_id=None,  #(batch,)  代表还在交互的序列编号  最大时是100
                    obs_next=None,  #[batch,1]  分批次的id值，只用于交互阶段
                    rew=None,  #(batch,)  每个对应的奖励值
                    done=None,
                    info=None,
                    policy=None,
                    dim_batch=None,
                    reset=False):
        if reset and dim_batch:   #self.data用于存储状态，len是最大长度，batch是同时运行的交互序列数，dim是状态维度，初始状态是用户转换得到的 
            self.data = torch.zeros(self.MAX_TURN, dim_batch, self.dim_model,
                                    device=self.device)  # (Length, Batch, Dim)
            self.len_data = torch.zeros(dim_batch, dtype=torch.int64)
            return

        res = {}

        if obs is not None:  # 1. initialize the state vectors
            if self.dataset == "VirtualTB-v0":
                e_u = self.get_embedding(obs[:, :-3], "user")
            else:  #elif self.dataset == "KuaishouEnv-v0":
                e_u = self.get_embedding(obs, "user")#[100,32]

            e_u_prime = self.ffn_user(e_u)

            length = 1
            self.len_data[env_id] = length
            self.data[0, env_id, :] = e_u_prime

            nowdata = self.data[:length, env_id, :]#[length, batch, dim]
            mask = torch.triu(torch.ones(length, length, device=self.device) * float('-inf'), diagonal=1)

            s0 = self.forward(nowdata, mask)#[batch,new_dim]

            res = {"obs": s0}


        elif obs_next is not None:  # 2. add action autoregressively
            if self.dataset == "VirtualTB-v0":
                a_t = self.get_embedding(obs_next[:, :-3], "action")
            else:  #elif self.dataset == "KuaishouEnv-v0":
                a_t = self.get_embedding(obs_next, "action")  #[batch,32]

            self.len_data[env_id] += 1
            length = int(self.len_data[env_id[0]])

            # turn = obs_next[:, -1]
            # assert all(self.len_data[env_id].numpy() == turn + 1)
            rew_matrix = rew.reshape((-1, 1))
            r_t = self.get_embedding(rew_matrix, "feedback")


            mask = torch.triu(torch.ones(length, length, device=self.device) * float('-inf'), diagonal=1)
            mask = mask
            
            au_t_past = self.fc1(self.data[length - 2, env_id, :])
            e_a = self.fc2(a_t)
            g_t = self.sigmoid(self.gate(torch.cat((e_a, au_t_past), -1)))
            au_t_prime = g_t * e_a + (1 - g_t) * au_t_past
            self.data[length - 1, env_id, :] = au_t_prime

            s_t = self.forward(self.data[:length, env_id, :], mask)  #[len, batch, 32] -> [len,20]

            res = {"obs_next": s_t}

        return res
        # return {"obs": obs, "env_id": env_id, "obs_next": obs_next, "rew": rew,
        #         "done": done, "info": info, "policy": policy}


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        if pe[:, 0, 1::2].shape[-1] % 2 == 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.adaptive_filter = adaptive_filter

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy < threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x):
        
        # 确保时间维度处于第 1 维
        x_in = x.permute(1, 0, 2)
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        
        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')   #dim=1
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')  #dim=1

        x = x.to(dtype)
        
        x = x.view(B, N, C)  # Reshape back to original shape
        x = x.permute(1, 0, 2)  # 重塑回 [seq_len, batch, dim]

        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(self.relu(x))
        x = self.fc2(x)
        return x


class Encode_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = PositionwiseFeedForward(dim, mlp_hidden_dim, drop)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.asb(self.norm1(x))

        return x

class Encoder(nn.Module):
    def __init__(self, dim_model,dropout,n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([Encode_layer(dim=dim_model,drop = dropout) for _ in range(n_layers)]
                                    )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class StateGRU(nn.Module):
    def __init__(self, dim_model, hidden_size, drop=0.1):
        super().__init__()

        self.asb = Adaptive_Spectral_Block(dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        

        self.gru = nn.GRU(
            input_size=dim_model,
            hidden_size=hidden_size,
            batch_first=False
        )
        # 定义线性层用于输出
        self.fc = nn.Linear(hidden_size, dim_model)

    def forward(self, x):
        
        x = self.asb(self.norm1(x)) 
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out)
        
        return out

if __name__ == '__main__':
   pass