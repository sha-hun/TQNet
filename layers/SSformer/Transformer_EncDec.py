import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, up_len, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.UpLayer = torch.nn.ModuleList(
            [
                nn.Linear(up_len[i], up_len[i - 1], bias=False)
                for i in range(1, len(up_len))
            ]
        )

        self.DownLayer = torch.nn.ModuleList(
            [
                nn.Linear(up_len[i], up_len[i + 1], bias=False)
                for i in range(0, len(up_len) - 1)
            ]
        )

    def forward(self, x, idx, attn_mask=None, tau=None, delta=None):
        results = []
        attns = []

        ql = []
        tq = x[-1].permute(0, 2, 1)
        for i in range(len(x) - 1, 0, -1):
            if i > 1 :
                for j in range(idx[i - 1], idx[i - 2], -1):
                    print("输入的形状是",tq.shape)
                    print("j层的",self.UpLayer[j].weight.shape)
                    tq = self.UpLayer[j](tq)

            else:
                for j in range(idx[i - 1], -1, -1):
                    tq = self.UpLayer[j](tq)
            tq = tq + x[i - 1].permute(0, 2, 1)
            ql.append(tq.permute(0, 2, 1))

        vl = []
        tv = x[0].permute(0, 2, 1)
        start = 0
        for i in range(0, len(x) - 1):
            for j in range(start, idx[i] + 1):
                tv = self.DownLayer[j](tv)
            start = idx[i] + 1
            tv = tv + x[i + 1].permute(0, 2, 1)
            vl.append(tv.permute(0, 2, 1))

        for i in range(len(x)):
            q, v = x[i], x[i]
            if i > 0:
                v = vl[i - 1]
            if i < len(x) - 1:
                q = ql[len(x) - 2 - i]

            new_x, attn = self.attention[i](q, x[i], v,
                                         attn_mask=attn_mask,
                                         tau=tau, delta=delta)
            x[i] = x[i] + self.dropout(new_x)
            y = x[i] = self.norm1(x[i])
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            results.append(self.norm2(x[i] + y))
            # results.append(x[i])
            attns.append(attn)

        return results, attns


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, idx, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, idx, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, idx, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            for i in range(len(x)):
                x[i] = self.norm(x[i])

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        results = []
        for i in range(len(x)):
            x[i] = x[i] + self.dropout(self.self_attention[i](
                x[i], x[i], x[i],
                attn_mask=x_mask,
                tau=tau, delta=None
            )[0])
            x[i] = self.norm1(x[i])

            x[i] = x[i] + self.dropout(self.cross_attention(
                x[i], cross[i], cross[i],
                attn_mask=cross_mask,
                tau=tau, delta=delta
            )[0])

            y = x[i] = self.norm2(x[i])
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            results.append(self.norm3(x[i] + y))

        return results


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            for i in range(len(x)):
                x[i] = self.norm(x[i])

        if self.projection is not None:
            for i in range(len(x)):
                x[i] = self.projection(x[i])
        return x
