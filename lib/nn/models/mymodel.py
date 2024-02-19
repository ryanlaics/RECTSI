import torch
from torch import nn
from lib.nn.models.Mytransformer import *


class PositionalEncodingModule(nn.Module):
    def __init__(self, size):
        super(PositionalEncodingModule, self).__init__()
        self.encoding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size)))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, -1)
        b, c, d, e = x.size()
        x = x.reshape(b,c,d*e)

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)

        y = x * y.expand_as(x)
        y = y.reshape(b,c,d,e)
        y = y.transpose(1, -1)
        return y

class MyImputer(nn.Module):

    def __init__(self, d_in):
        super(MyImputer, self).__init__()
        self.num_nodes = d_in
        if d_in == 36:
            self.steps_per_day = 24
            # self.window_size = 36
            # self.window_size = 24
            self.window_size = 24
        if d_in == 64:
            self.steps_per_day = 288
            self.window_size = 24
        if d_in == 56:
            self.steps_per_day = 1
            self.window_size = 14
            print('covid')

        self.base_dim = 8
        self.groups = 2
        self.num_layers = 1

        self.input_emb_dim = self.base_dim * 2
        self.tod_emb_dim = self.base_dim // 2
        self.dow_emb_dim = self.base_dim // 2
        self.spatial_emb_dim = self.base_dim
        # self.spatial_emb_dim = 0
        # self.adp_emb_dim = self.base_dim * 2 - 1
        # self.adp_emb_dim = self.base_dim * 4 - 1
        self.adp_emb_dim = self.base_dim * 2 - 1


        self.model_dim = (
                self.input_emb_dim
                + self.tod_emb_dim
                + self.dow_emb_dim
                + self.spatial_emb_dim
                + self.adp_emb_dim
                + 1
        )
        self.model_dim2= (self.model_dim-self.input_emb_dim)//2 + self.input_emb_dim
        print('self.window_size', self.window_size)
        print('self.input_emb_dim',self.input_emb_dim)
        print('self.model_dim',self.model_dim)
        print('self.model_dim2',self.model_dim2)
        print('self.base_dim', self.base_dim)
        print('self.num_layers', self.num_layers)
        self.input_proj = nn.Linear(3, self.input_emb_dim)
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_emb_dim)
        self.dow_embedding = nn.Embedding(7, self.dow_emb_dim)


        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_emb_dim))
        self.adp_emb = nn.Parameter(torch.empty(self.window_size, self.num_nodes, self.adp_emb_dim))

        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.adp_emb)
        self.output_proj = nn.Linear(self.window_size * self.model_dim2, self.window_size)

        # self.attn_layers_t = nn.ModuleList([
        #     SelfAttentionLayer(self.model_dim2, groups = self.groups)
        #     for _ in range(self.num_layers)])

        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(self.model_dim2, groups = self.groups)
            for _ in range(self.num_layers)])

        # Temporal Positional Encoding
        # self.temporal_positional_encoding = nn.ModuleList([
        #     PositionalEncodingModule((self.window_size, 1, self.model_dim2))
        #     for _ in range(self.num_layers)
        # ])

        # Spatial Positional Encoding
        self.spatial_positional_encoding = nn.ModuleList([
            PositionalEncodingModule((1, self.num_nodes, self.model_dim2))
            for _ in range(self.num_layers)
        ])

        self.se = SELayer(self.model_dim-self.input_emb_dim-1)
        print('self.model_dim-self.input_emb_dim-1',self.model_dim-self.input_emb_dim-1)
        self.compressed_layer = nn.Linear(self.model_dim-self.input_emb_dim-1, (self.model_dim-self.input_emb_dim)//2-1)
        print('(self.model_dim-self.input_emb_dim)//2-1',(self.model_dim-self.input_emb_dim)//2-1)

        # print('no skip')
        print(' skip')

    def forward(self, x, mask, adj=None, pos=None, timestamp=None, adj_label=None):
        init_x = x
        third_of_last_dim = x.shape[-1] // 3

        x = x[..., :third_of_last_dim]

        # masked_x = torch.where(mask, x, torch.zeros_like(x))
        masked_x = x

        input_data_1 = masked_x.unsqueeze(-1)
        input_data_2 = init_x[..., third_of_last_dim:third_of_last_dim * 2].unsqueeze(-1)
        input_data_3 = init_x[..., third_of_last_dim * 2:].unsqueeze(-1)

        x = torch.cat([input_data_1, input_data_2, input_data_3], dim=-1)
        batch_size = x.shape[0]

        tod = x[..., 1]
        dow = x[..., 2]
        x = x[..., :3]


        x = self.input_proj(x)  # (batch_size, window_size, num_nodes, input_emb_dim)
        features = []
        tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
        # print('tod_emb',tod_emb.size())
        features.append(tod_emb)
        dow_emb = self.dow_embedding((dow).long())
        # print('dow_emb',dow_emb.size())
        features.append(dow_emb)
        spatial_emb = self.node_emb.expand(batch_size, self.window_size, *self.node_emb.shape)
        # print('spatial_emb',spatial_emb.size())
        features.append(spatial_emb)
        adp_emb = self.adp_emb.expand(size=(batch_size, *self.adp_emb.shape))
        # print('adp_emb',adp_emb.size())
        features.append(adp_emb)


        total_embedding = torch.cat(features, dim=-1)

        total_embedding = self.se(total_embedding)
        total_embedding = self.compressed_layer(total_embedding)

        x = torch.cat([x,total_embedding,mask.unsqueeze(-1)], dim=-1)

        for i in range(self.num_layers):
            temp_x = x
            # x = x + self.temporal_positional_encoding[i].encoding
            # x = self.attn_layers_t[i](x, dim=1)
            x = x + self.spatial_positional_encoding[i].encoding
            x = self.attn_layers_s[i](x, dim=2)
            x = x + temp_x


        out = x.transpose(1, 2)
        out = out.reshape(
            batch_size, self.num_nodes, self.window_size * self.model_dim2
        )
        out = self.output_proj(out).view(
            batch_size, self.num_nodes, self.window_size
        )
        out = out.transpose(1, 2)
        # final_output = torch.where(mask, masked_x, out)
        final_output = out
        return final_output, final_output

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-in', type=int)
        return parser



# import torch
# from torch import nn
# from lib.nn.models.Mytransformer import *
#
#
# class PositionalEncodingModule(nn.Module):
#     def __init__(self, size):
#         super(PositionalEncodingModule, self).__init__()
#         self.encoding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size)))
#
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=8):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = x.transpose(1, -1)
#         b, c, d, e = x.size()
#         x = x.reshape(b,c,d*e)
#
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1)
#
#         y = x * y.expand_as(x)
#         y = y.reshape(b,c,d,e)
#         y = y.transpose(1, -1)
#         return y
#
# class MyImputer(nn.Module):
#
#     def __init__(self, d_in):
#         super(MyImputer, self).__init__()
#         self.num_nodes = d_in
#         if d_in == 36:
#             self.steps_per_day = 24
#             # self.window_size = 36
#             self.window_size = 24
#         if d_in == 64:
#             self.steps_per_day = 288
#             self.window_size = 24
#         if d_in == 56:
#             self.steps_per_day = 1
#             self.window_size = 14
#             print('covid')
#
#         self.base_dim = 8
#         self.groups = 1
#         self.num_layers = 1
#
#         self.input_emb_dim = self.base_dim * 2
#         self.tod_emb_dim = self.base_dim // 2
#         self.dow_emb_dim = self.base_dim // 2
#         self.spatial_emb_dim = self.base_dim
#         self.adp_emb_dim = self.base_dim * 2 - 1
#
#         self.model_dim = (
#                 self.input_emb_dim
#                 + self.tod_emb_dim
#                 + self.dow_emb_dim
#                 + self.spatial_emb_dim
#                 + self.adp_emb_dim
#                 + 1
#         )
#         self.model_dim2= (self.model_dim-self.input_emb_dim)//2 + self.input_emb_dim
#
#         print('self.input_emb_dim',self.input_emb_dim)
#         print('self.model_dim',self.model_dim)
#         print('self.model_dim2',self.model_dim2)
#         print('self.base_dim', self.base_dim)
#         print('self.num_layers', self.num_layers)
#         # print('self.groups', self.groups)
#         self.input_proj = nn.Linear(3, self.input_emb_dim)
#         self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_emb_dim)
#         self.dow_embedding = nn.Embedding(7, self.dow_emb_dim)
#
#
#         self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.spatial_emb_dim))
#         self.adp_emb = nn.Parameter(torch.empty(self.window_size, self.num_nodes, self.adp_emb_dim))
#
#         nn.init.xavier_uniform_(self.node_emb)
#         nn.init.xavier_uniform_(self.adp_emb)
#         self.output_proj = nn.Linear(self.window_size * self.model_dim2, self.window_size)
#
#         self.attn_layers_t = nn.ModuleList([
#             SelfAttentionLayer(self.model_dim2, groups = self.groups)
#             for _ in range(self.num_layers)])
#
#         self.attn_layers_s = nn.ModuleList([
#             SelfAttentionLayer(self.model_dim2, groups = self.groups)
#             for _ in range(self.num_layers)])
#
#         # Temporal Positional Encoding
#         self.temporal_positional_encoding = nn.ModuleList([
#             PositionalEncodingModule((self.window_size, 1, self.model_dim2))
#             for _ in range(self.num_layers)
#         ])
#
#         # Spatial Positional Encoding
#         self.spatial_positional_encoding = nn.ModuleList([
#             PositionalEncodingModule((1, self.num_nodes, self.model_dim2))
#             for _ in range(self.num_layers)
#         ])
#
#         self.se = SELayer(self.model_dim-self.input_emb_dim-1)
#         self.compressed_layer = nn.Linear(self.model_dim-self.input_emb_dim-1, (self.model_dim-self.input_emb_dim)//2-1)
#
#         # print('no skip')
#         print(' skip')
#
#     def forward(self, x, mask, adj=None, pos=None, timestamp=None, adj_label=None):
#         init_x = x
#         third_of_last_dim = x.shape[-1] // 3
#
#         x = x[..., :third_of_last_dim]
#
#         masked_x = torch.where(mask, x, torch.zeros_like(x))
#
#         input_data_1 = masked_x.unsqueeze(-1)
#         input_data_2 = init_x[..., third_of_last_dim:third_of_last_dim * 2].unsqueeze(-1)
#         input_data_3 = init_x[..., third_of_last_dim * 2:].unsqueeze(-1)
#
#         x = torch.cat([input_data_1, input_data_2, input_data_3], dim=-1)
#         batch_size = x.shape[0]
#
#         tod = x[..., 1]
#         dow = x[..., 2]
#         x = x[..., :3]
#
#
#         x = self.input_proj(x)  # (batch_size, window_size, num_nodes, input_emb_dim)
#         features = []
#         tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
#         features.append(tod_emb)
#         dow_emb = self.dow_embedding((dow).long())
#         features.append(dow_emb)
#         spatial_emb = self.node_emb.expand(batch_size, self.window_size, *self.node_emb.shape)
#         features.append(spatial_emb)
#         adp_emb = self.adp_emb.expand(size=(batch_size, *self.adp_emb.shape))
#         features.append(adp_emb)
#
#
#         total_embedding = torch.cat(features, dim=-1)
#
#         total_embedding = self.se(total_embedding)
#         total_embedding = self.compressed_layer(total_embedding)
#
#         x = torch.cat([x,total_embedding,mask.unsqueeze(-1)], dim=-1)
#
#         for i in range(self.num_layers):
#             temp_x = x
#             x = x + self.temporal_positional_encoding[i].encoding
#             x = self.attn_layers_t[i](x, dim=1)
#             x = x + self.spatial_positional_encoding[i].encoding
#             x = self.attn_layers_s[i](x, dim=2)
#             x = x + temp_x
#
#
#         out = x.transpose(1, 2)
#         out = out.reshape(
#             batch_size, self.num_nodes, self.window_size * self.model_dim2
#         )
#         out = self.output_proj(out).view(
#             batch_size, self.num_nodes, self.window_size
#         )
#         out = out.transpose(1, 2)
#         final_output = torch.where(mask, masked_x, out)
#         return final_output, final_output
#
#     @staticmethod
#     def add_model_specific_args(parser):
#         parser.add_argument('--d-in', type=int)
#         return parser
#
#
