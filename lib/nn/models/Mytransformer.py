import torch
from torch import nn


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, height, width, num_channels = x.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batch_size, height, width, self.groups, channels_per_group)

        # transpose
        x = x.permute(0, 3, 1, 2, 4).contiguous()

        # flatten
        x = x.view(batch_size, height, width, -1)

        return x


class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=4, groups=1):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.head_dim = self.model_dim // self.num_heads

        self.FC_Q = nn.Linear(model_dim // groups, model_dim // groups)
        self.FC_K = nn.Linear(model_dim // groups, model_dim // groups)
        self.FC_V = nn.Linear(model_dim // groups, model_dim // groups)
        self.groups = groups
        self.channel_shuffle = ChannelShuffle(groups)

        self.out_proj = nn.Linear(model_dim // groups, model_dim // groups)

    def forward(self, query, key, value, mask=None):
        input_shape = query.shape
        batch_size = query.shape[0]

        if self.groups != 1:
            query = self.FC_Q(query.view(input_shape[0], input_shape[1], input_shape[2], self.groups, -1)).view(
                input_shape[0], input_shape[1], input_shape[2], input_shape[3])
            key = self.FC_K(key.view(input_shape[0], input_shape[1], input_shape[2], self.groups, -1)).view(
                input_shape[0], input_shape[1], input_shape[2], input_shape[3])
            value = self.FC_V(value.view(input_shape[0], input_shape[1], input_shape[2], self.groups, -1)).view(
                input_shape[0], input_shape[1], input_shape[2], input_shape[3])

            query = self.channel_shuffle(query)
            key = self.channel_shuffle(key)
            value = self.channel_shuffle(value)
        else:
            query = self.FC_Q(query)
            key = self.FC_K(key)
            value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if mask is not None:
            multihead_mask = mask.repeat(self.num_heads, 1, 1, 1)
            attn_score.masked_fill_(multihead_mask == 0, -1e8)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out.view(input_shape[0], input_shape[1], input_shape[2], self.groups, -1)).view(
            input_shape[0], input_shape[1], input_shape[2], input_shape[3])

        # out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=256, num_heads=4, dropout=0.1, groups=1
    ):
        super().__init__()
        self.groups = 1
        print('ffn group', self.groups)
        self.attn = AttentionLayer(model_dim, num_heads, groups)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim // self.groups, feed_forward_dim // self.groups),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim // self.groups, model_dim // self.groups),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x, mask)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        input_shape = out.shape
        out = self.feed_forward(out.view(input_shape[0], input_shape[1], input_shape[2], self.groups, -1)).view(
            input_shape[0], input_shape[1], input_shape[2], input_shape[3])

        # out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out