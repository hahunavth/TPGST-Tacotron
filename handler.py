from typing import  Dict, List, Any
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import os

from config import ConfigArgs as args

_punctuation = "!'(),.:;?"
_letters = "t aw1 aa6 nc e3 wa1 iz4 oz6 uw6 e1 oz4 oz3 mc u1 ee1 aa5 p ooo3 kh aa1 uw5 ie1 o4 w o2 tr chz iz6 uw3 ow5 e4 m e5 a1 uw2 uw1 a3 uz1 wa4 ee5 b nh iz3 h ng oz1 o3 oo4 i4 uo1 uo5 nhz ee6 i6 i3 th iz1 ph aw2 a6 uo2 uo6 a5 aa2 v a2 wa3 o6 g x ch oo5 ie2 oo1 kc ie3 ie4 ow4 aw4 aw3 k ooo2 ooo1 uz3 a4 ow3 oo3 uw4 aa3 o5 iz2 aw6 i5 uz6 n wa5 i1 ooo5 uz2 ee4 ow2 ee2 ow6 wa2 e2 d iz5 u5 pc r u3 aa4 ngz u4 uz5 u2 uz4 l oo2 aw5 oz2 tc u6 ooo6 ee3 oo6 ie5 dd uo3 uo4 oz5 ow1 ie6 i2 wa6 ooo4 e6 o1"
_letters = _letters.split()

_silences = ["spn","sil", "sp"]
_space = ["spc","dot"]

# Export all symbols:
symbols = (
    _space
    + list(_punctuation)
    + _silences
    + _letters
)

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def load_vocab():
    """
    Makes dictionaries

    Returns:
        :char2idx: Dictionary containing characters as keys and corresponding indexes as values
        :idx2char: Dictionary containing indexes as keys and corresponding characters as values

    """
    return _symbol_to_id, _id_to_symbol


def text_collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts,).

    :param data: list of tuple (texts,).

    Returns:
        text_pads: torch tensor of shape (batch_size, padded_length).

    """
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    texts, _ = zip(*data)

    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    for idx in range(len(texts)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
    return text_pads, None, None



class Conv1d(nn.Conv1d):
    """
    :param in_channels: Scalar
    :param out_channels: Scalar
    :param kernel_size: Scalar
    :param activation_fn: activation function
    :param drop_rate: Scalar. dropout rate
    :param stride: Scalar
    :param padding: padding type
    :param dilation: Scalar
    :param groups: Scalar
    :param bias: Boolean.
    :param bn: Boolean. whether it uses batch normalization

    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True, bn=False):
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate
        if padding == 'same':
            padding = kernel_size // 2 * dilation
            self.even_kernel = not bool(kernel_size % 2)
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)
        self.drop_out = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.001) if bn else None

    def forward(self, x):
        """
        :param x: (N, C_in, T) Tensor.

        Returns:
            y: (N, C_out, T) Tensor.

        """
        y = super(Conv1d, self).forward(x)
        y = self.batch_norm(y) if self.batch_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        y = self.drop_out(y) if self.drop_out is not None else y
        y = y[:, :, :-1] if self.even_kernel else y
        return y


class Conv1dBank(nn.Module):
    """
    :param in_channels: Scalar.
    :param out_channels: Scalar.
    :param K: Scalar. K sets for 1-d convolutional filters
    :param activation_fn: activation function

    """

    def __init__(self, in_channels, out_channels, K, activation_fn=None):
        self.K = K
        super(Conv1dBank, self).__init__()
        self.conv_bank = nn.ModuleList([
            Conv1d(in_channels, out_channels, k, activation_fn=activation_fn, bias=False, bn=True)
            for k in range(1, self.K + 1)
        ])

    def forward(self, x):
        """
        :param x: (N, C_in, T) Tensor.

        Returns:
            y: (N, K*C_out, T) Tensor.

        """
        convs = []
        for i in range(self.K):
            convs.append(self.conv_bank[i](x))
        y = torch.cat(convs, dim=1)
        return y


class Highway(nn.Linear):
    """
    :param input_dim: Scalar.
    :param drop_rate: Scalar. dropout rate

    """

    def __init__(self, input_dim, drop_rate=0.):
        self.drop_rate = drop_rate
        super(Highway, self).__init__(input_dim, input_dim * 2)
        self.drop_out = nn.Dropout(self.drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        """
        :param x: (N, T, input_dim) Tensor.

        Returns:
            y: (N, T, input_dim) Tensor.

        """
        y = super(Highway, self).forward(x)  # (N, C_out*2, T)
        h, y_ = y.chunk(2, dim=-1)  # half size for axis C_out. (N, C_out, T) respectively
        h = torch.sigmoid(h)  # Gate
        y_ = torch.relu(y_)
        y_ = h * y_ + (1 - h) * x
        y_ = self.drop_out(y_) if self.drop_out is not None else y_
        return y_


class HighwayConv1d(Conv1d):
    """
    :param in_channels: Scalar
    :param out_channels: Scalar
    :param kernel_size: Scalar
    :param drop_rate: Scalar. dropout rate
    :param stride: Scalar
    :param padding: padding type
    :param dilation: Scalar
    :param groups: Scalar
    :param bias: Boolean.

    """

    def __init__(self, in_channels, out_channels, kernel_size, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True):
        self.drop_rate = drop_rate
        super(HighwayConv1d, self).__init__(in_channels, out_channels * 2, kernel_size, activation_fn=None,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)
        self.drop_out = nn.Dropout(self.drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        """
        :param x: (N, C_in, T) Tensor.

        Returns:
            y: (N, C_out, T) Tensor.

        """
        y = super(HighwayConv1d, self).forward(x)  # (N, C_out*2, T)
        h, y_ = y.chunk(2, dim=1)  # half size for axis C_out. (N, C_out, T) respectively
        h = torch.sigmoid(h)  # Gate
        y_ = torch.relu(y_)
        y_ = h * y_ + (1 - h) * x
        y_ = self.drop_out(y_) if self.drop_out is not None else y_
        return y_


class AttentionRNN(nn.Module):
    """
    :param enc_dim: Scalar.
    :param dec_dim: Scalar.

    """

    def __init__(self, enc_dim, dec_dim):
        super(AttentionRNN, self).__init__()
        self.gru = nn.GRU(dec_dim, dec_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.att = BilinearAttention(enc_dim, dec_dim)

    def forward(self, h, s, prev_hidden=None):
        """
        :param h: (N, Tx, enc_dim) Tensor. Encoder outputs
        :param s: (N, Ty/r, dec_dim) Tensor. Decoder inputs (previous decoder outputs)

        Returns:
            :s: (N, Ty/r, dec_dim) Tensor. Decoder outputs
            :A: (N, Ty/r, Tx) Tensor. Attention
            :hidden: Tensor.
        """
        # Attention RNN
        s, hidden = self.gru(s, prev_hidden)  # (N, Ty/r, Cx)
        A = self.att(h, s)  # (N, Ty/r, Tx)
        return s, A, hidden


class MLPAttention(nn.Module):
    """
    :param enc_dim: Scalar.
    :param dec_dim: Scalar.

    """

    def __init__(self, enc_dim, dec_dim):
        super(MLPAttention, self).__init__()
        self.W = nn.Linear(enc_dim + dec_dim, args.Ca, bias=True)
        self.v = nn.Linear(args.Ca, 1, bias=False)

    def forward(self, h, s):
        """
        :param h: (N, Tx, Cx) Tensor. Encoder outputs
        :param s: (N, Ty/r, Cx) Tensor. Decoder inputs (previous decoder outputs)

        Returns:
            A: (N, Ty/r, Tx) Tensor. attention

        """
        Tx, Ty = h.size(1), s.size(1)  # Tx, Ty
        hs = torch.cat([h.unsqueeze(1).expand(-1, Ty, -1, -1), s.unsqueeze(2).expand(-1, -1, Tx, -1)], dim=-1)
        e = self.v(torch.tanh(self.W(hs))).squeeze(-1)
        A = torch.softmax(e, dim=-1)
        return A


class BilinearAttention(nn.Module):
    """
    :param enc_dim: Scalar.
    :param dec_dim: Scalar

    """

    def __init__(self, enc_dim, dec_dim):
        super(BilinearAttention, self).__init__()
        self.W = nn.Linear(enc_dim, dec_dim)

    def forward(self, h, s):
        """
        :param h: (N, Tx, Cx) Tensor. Encoder outputs
        :param s: (N, Ty/r, Cx) Tensor. Decoder inputs (previous decoder outputs)

        Returns:
            A: (N, Ty/r, Tx) Tensor. attention

        """
        wh = self.W(h)  # (N, Tx, Es)
        e = torch.matmul(wh, s.transpose(1, 2))  # (N, Tx, Ty)
        A = torch.softmax(e.transpose(1, 2), dim=-1)  # (N, Ty, Tx)
        return A


class Conv1d(nn.Conv1d):
    """
    :param in_channels: Scalar
    :param out_channels: Scalar
    :param kernel_size: Scalar
    :param activation_fn: activation function
    :param drop_rate: Scalar. dropout rate
    :param stride: Scalar
    :param padding: padding type
    :param dilation: Scalar
    :param groups: Scalar
    :param bias: Boolean.
    :param bn: Boolean. whether it uses batch normalization

    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True, bn=False):
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate
        if padding == 'same':
            padding = kernel_size // 2 * dilation
            self.even_kernel = not bool(kernel_size % 2)
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)
        self.drop_out = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.001) if bn else None

    def forward(self, x):
        """
        :param x: (N, C_in, T) Tensor.

        Returns:
            y: (N, C_out, T) Tensor.

        """
        y = super(Conv1d, self).forward(x)
        y = self.batch_norm(y) if self.batch_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        y = self.drop_out(y) if self.drop_out is not None else y
        y = y[:, :, :-1] if self.even_kernel else y
        return y


class PreNet(nn.Module):
    """

    :param input_dim: Scalar.
    :param hidden_dim: Scalar.

    """

    def __init__(self, input_dim, hidden_dim):
        super(PreNet, self).__init__()
        self.prenet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        """
        :param x: (N, Tx, Ce) Tensor.

        Returns:
            y_: (N, Tx, Cx) Tensor.

        """
        y_ = self.prenet(x)
        return y_



class Conv2d(nn.Conv2d):
    """
    :param in_channels: Scalar
    :param out_channels: Scalar
    :param kernel_size: Scalar
    :param activation_fn: activation function
    :param drop_rate: Scalar. dropout rate
    :param stride: Scalar
    :param padding: padding type
    :param dilation: Scalar
    :param groups: Scalar.
    :param bias: Boolean.
    :param bn: Boolean. whether it uses batch normalization

    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True, bn=False):
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate
        if padding == 'same':
            padding = kernel_size // 2 * dilation
            self.even_kernel = not bool(kernel_size % 2)
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias)
        self.drop_out = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.001) if bn else None

    def forward(self, x):
        """
        :param x: (N, C_in, T) Tensor.

        Returns:
            y: (N, C_out, T) Tensor.
        """
        y = super(Conv2d, self).forward(x)
        y = self.batch_norm(y) if self.batch_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        y = self.drop_out(y) if self.drop_out is not None else y
        y = y[:, :, :-1] if self.even_kernel else y
        return y


class CBHG(nn.Module):
    """
    CBHG module (Convolution bank + Highwaynet + GRU)

    :param input_dim: Scalar.
    :param hidden_dim: Scalar.
    :param K: Scalar. K sets of 1-D conv filters
    :param n_highway: Scalar. number of highway layers
    :param bidirectional: Boolean. whether it is bidirectional

    """

    def __init__(self, input_dim, hidden_dim, K=16, n_highway=4, bidirectional=True):
        super(CBHG, self).__init__()
        self.K = K
        self.conv_bank = Conv1dBank(input_dim, hidden_dim, K=self.K, activation_fn=torch.relu)
        self.max_pool = nn.MaxPool1d(2, stride=1, padding=1)
        self.projection = nn.Sequential(
            Conv1d(self.K * hidden_dim, hidden_dim, 3, activation_fn=torch.relu, bias=False, bn=True),
            Conv1d(hidden_dim, input_dim, 3, bias=False, bn=True),
        )
        self.highway = nn.ModuleList(
            [Highway(input_dim) for _ in range(n_highway)]
        )
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True,
                          bidirectional=bidirectional)  # if batch_first is True, (Batch, Sequence, Feature)

    def forward(self, x, prev=None):
        """
        :param x: (N, T, input_dim) Tensor.
        :param prev: Tensor. for gru

        Returns:
            :y_: (N, T, 2*hidden_dim) Tensor.
            :hidden: Tensor. hidden state

        """
        y_ = x.transpose(1, 2)  # (N, input_dim, Tx)
        y_ = self.conv_bank(y_)  # (N, K*hidden_dim, Tx)
        y_ = self.max_pool(y_)[:, :, :-1]  # pooling over time
        y_ = self.projection(y_)  # (N, input_dim, Tx)
        y_ = y_.transpose(1, 2)  # (N, Tx, input_dim)
        # Residual connection
        y_ = y_ + x  # (N, Tx, input_dim)
        for idx in range(len(self.highway)):
            y_ = self.highway[idx](y_)  # (N, Tx, input_dim)
        y_, hidden = self.gru(y_, prev)  # (N, Tx, hidden_dim)
        return y_, hidden


class TextEncoder(nn.Module):
    """
    Text Encoder
    Prenet + CBHG

    """

    def __init__(self, hidden_dims):
        super(TextEncoder, self).__init__()
        self.prenet = PreNet(args.Ce, hidden_dims)
        self.cbhg = CBHG(input_dim=hidden_dims, hidden_dim=hidden_dims, K=16, n_highway=4, bidirectional=True)

    def forward(self, x):
        """
        :param x: (N, Tx, Ce) Tensor. Character embedding

        Returns:
            :y_: (N, Tx, 2*Cx) Text Embedding
            :hidden: Tensor.

        """
        y_ = self.prenet(x)  # (N, Tx, Cx)
        y_, hidden = self.cbhg(y_)  # (N, Cx*2, Tx)
        return y_, hidden


class ReferenceEncoder(nn.Module):
    """
    Reference Encoder.
    6 convs + GRU + FC

    :param in_channels: Scalar.
    :param embed_size: Scalar.
    :param activation_fn: activation function

    """

    def __init__(self, in_channels=1, embed_size=128, activation_fn=None):
        super(ReferenceEncoder, self).__init__()
        self.embed_size = embed_size
        channels = [in_channels, 32, 32, 64, 64, 128, embed_size]
        self.convs = nn.ModuleList([
            Conv2d(channels[c], channels[c + 1], 3, stride=2, bn=True, bias=False, activation_fn=torch.relu)
            for c in range(len(channels) - 1)
        ])  # (N, Ty/r, 128)
        self.gru = nn.GRU(self.embed_size * 2, self.embed_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size),
        )
        self.activation_fn = activation_fn

    def forward(self, x, hidden=None):
        """
        :param x: (N, 1, Ty, n_mels) Tensor. Mel Spectrogram
        :param hidden: Tensor. initial hidden state for gru

        Returns:
            y_: (N, 1, E) Reference Embedding

        """
        y_ = x
        for i in range(len(self.convs)):
            y_ = self.convs[i](y_)
        # (N, C, Ty//64, n_mels//64)
        y_ = y_.transpose(1, 2)  # (N, Ty//64, C, n_mels//64)
        shape = y_.shape
        y_ = y_.contiguous().view(shape[0], -1, shape[2] * shape[3])  # (N, Ty//64, C*n_mels//64)
        y_, out = self.gru(y_, hidden)  # (N, Ty//64, E)
        # y_ = y_[:, -1, :] # (N, E)
        y_ = out.squeeze(0)  # same as states[:, -1, :]
        y_ = self.fc(y_)  # (N, E)
        y_ = self.activation_fn(y_) if self.activation_fn is not None else y_
        return y_.unsqueeze(1)


class StyleTokenLayer(nn.Module):
    """
    Style Token Layer
    Reference Encoder + Multi-head Attention, token embeddings

    :param embed_size: Scalar.
    :param n_units: Scalar. for multihead attention ***

    """

    def __init__(self, embed_size=128, n_units=128):
        super(StyleTokenLayer, self).__init__()
        self.token_embedding = nn.Parameter(torch.zeros([args.n_tokens, embed_size]))  # (n_tokens, E)
        self.ref_encoder = ReferenceEncoder(in_channels=1, embed_size=embed_size, activation_fn=torch.tanh)
        self.att = MultiHeadAttention(n_units, embed_size)

        torch.nn.init.normal_(self.token_embedding, mean=0., std=0.5)
        # init.orthogonal_(self.token_embedding)

    def forward(self, ref, ref_mode=True):
        """
        :param ref: (N, Ty, n_mels) Tensor containing reference audio or (N, n_tokens) if not ref_mode
        :param ref_mode: Boolean. whether it is reference mode

        Returns:
            :y_: (N, 1, E) Style embedding
            :A: (N, n_tokens) Tensor. Combination weight.

        """
        token_embedding = self.token_embedding.unsqueeze(0).expand(ref.size(0), -1, -1)  # (N, n_tokens, E)
        if ref_mode:
            ref = self.ref_encoder(ref)  # (N, 1, E)
            A = torch.softmax(self.att(ref, token_embedding), dim=-1)  # (N, n_tokens)
            # A = torch.softmax(self.att(ref, token_embedding)) # (N, n_tokens)
        else:
            A = torch.softmax(ref, dim=-1)
        y_ = torch.sum(A.unsqueeze(-1) * token_embedding, dim=1, keepdim=True)  # (N, 1, E)
        y_ = torch.tanh(y_)
        return y_, A


class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention

    :param n_units: Scalars.
    :param embed_size : Scalars.

    """

    def __init__(self, n_units=128, embed_size=128):
        super(MultiHeadAttention, self).__init__()
        self.split_size = n_units // args.n_heads
        self.conv_Q = Conv1d(embed_size, n_units, 1)
        self.conv_K = Conv1d(embed_size, n_units, 1)
        self.fc_Q = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.Tanh(),
        )
        self.fc_K = nn.Sequential(
            nn.Linear(n_units, n_units),
            nn.Tanh(),
        )
        self.fc_V = nn.Sequential(
            nn.Linear(embed_size, self.split_size),
            nn.Tanh(),
        )
        self.fc_A = nn.Sequential(
            nn.Linear(n_units, args.n_tokens),
            nn.Tanh(),
        )

    def forward(self, ref_embedding, token_embedding):
        """
        :param ref_embedding: (N, 1, E) Reference embedding
        :param token_embedding: (N, n_tokens, embed_size) Token Embedding

        Returns:
            y_: (N, n_tokens) Tensor. Style attention weight

        """
        Q = self.fc_Q(self.conv_Q(ref_embedding.transpose(1, 2)).transpose(1, 2))  # (N, 1, n_units)
        K = self.fc_K(self.conv_K(token_embedding.transpose(1, 2)).transpose(1, 2))  # (N, n_tokens, n_units)
        V = self.fc_V(token_embedding)  # (N, n_tokens, n_units)
        Q = torch.stack(Q.split(self.split_size, dim=-1), dim=0)  # (n_heads, N, 1, n_units//n_heads)
        K = torch.stack(K.split(self.split_size, dim=-1), dim=0)  # (n_heads, N, n_tokens, n_units//n_heads)
        V = torch.stack(V.split(self.split_size, dim=-1), dim=0)  # (n_heads, N, n_tokens, n_units//n_heads)
        inner_A = torch.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / self.split_size ** 0.5,
            dim=-1
        )  # (n_heads, N, 1, n_tokens)
        y_ = torch.matmul(inner_A, V)  # (n_heads, N, 1, n_units//n_heads)
        y_ = torch.cat(y_.split(1, dim=0), dim=-1).squeeze()  # (N, n_units)
        y_ = self.fc_A(y_)  # (N, n_tokens)
        return y_


class AudioDecoder(nn.Module):
    """
    Audio Decoder
    prenet + attention RNN + 2 RNN + FC + CBHG

    :param enc_dim: Scalar. for encoder output
    :param dec_dim: Scalar. for decoder input

    """

    def __init__(self, enc_dim, dec_dim):
        super(AudioDecoder, self).__init__()
        self.prenet = PreNet(args.n_mels * args.r, dec_dim)
        self.attention_rnn = AttentionRNN(enc_dim=enc_dim, dec_dim=dec_dim)
        self.proj_att = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.decoder_rnn = nn.ModuleList([
            nn.GRU(dec_dim, dec_dim, num_layers=1, batch_first=True, bidirectional=False)
            for _ in range(2)
        ])
        self.final_frame = nn.Sequential(
            nn.Linear(dec_dim, 1),
            nn.Sigmoid(),
        )
        self.proj_mel = nn.Linear(dec_dim, args.n_mels * args.r)
        self.cbhg = CBHG(input_dim=args.n_mels, hidden_dim=dec_dim // 2, K=8, n_highway=4, bidirectional=True)
        self.proj_mag = nn.Linear(dec_dim, args.n_mels)

    def forward(self, decoder_inputs, encoder_outputs, prev_hidden=None, synth=False):
        """
        :param decoder_inputs: (N, Ty/r, n_mels*r) Tensor. Decoder inputs (previous decoder outputs)
        :param encoder_outputs: (N, Tx, Cx) Tensor. Encoder output *** general??
        :param prev_hidden: Tensor. hidden state for gru when synth is true
        :param synth: Boolean. whether it synthesizes

        Returns:
            :mels_hat: (N, Ty/r, n_mels*r) Mel spectrogram
            :mags_hat: (N, Ty, n_mags) Magnitude spectrogram
            :A: (N, Ty/r, Tx) Tensor. Attention weights
            :ff_hat: (N, Ty/r, 1) Tensor. for binary final frame prediction

        """
        if not synth:  # Train mode & Eval mode
            y_ = self.prenet(decoder_inputs)  # (N, Ty/r, Cx)
            # Attention RNN
            y_, A, hidden = self.attention_rnn(encoder_outputs, y_)  # y_: (N, Ty/r, Cx), A: (N, Ty/r, Tx)
            # (N, Ty/r, Tx) . (N, Tx, Cx)
            c = torch.matmul(A, encoder_outputs)  # (N, Ty/r, Cx)
            y_ = self.proj_att(torch.cat([c, y_], dim=-1))  # (N, Ty/r, Cx)

            # Decoder RNN
            for idx in range(len(self.decoder_rnn)):
                y_f, _ = self.decoder_rnn[idx](y_)  # (N, Ty/r, Cx)
                y_ = y_ + y_f

            # binary final frame prediction
            ff_hat = torch.clamp(self.final_frame(y_) + 1e-10, 1e-10, 1)  # (N, Ty/r, 1)

            # Mel-spectrogram
            mels_hat = self.proj_mel(y_)  # (N, Ty/r, n_mels*r)

            # Decoder CBHG
            y_ = mels_hat.view(mels_hat.size(0), -1, args.n_mels)  # (N, Ty, n_mels)
            y_, _ = self.cbhg(y_)  # (N, Ty, Cx*2)
            mags_hat = self.proj_mag(y_)  # (N, Ty, n_mags)
            return mels_hat, mags_hat, A, ff_hat
        else:
            # decoder_inputs: GO frame (N, 1, n_mels*r)
            att_hidden = None
            dec_hidden = [None, None]

            mels_hat = []
            mags_hat = []
            attention = []
            for idx in range(args.max_Ty):
                y_ = self.prenet(decoder_inputs)  # (N, 1, Cx)
                # Attention RNN
                y_, A, att_hidden = self.attention_rnn(encoder_outputs, y_, prev_hidden=att_hidden)
                attention.append(A)
                # Encoder outputs: (N, Tx, Cx)
                # A: (N, )
                c = torch.matmul(A, encoder_outputs)  # (N, Ty/r, Cx)
                y_ = self.proj_att(torch.cat([c, y_], dim=-1))  # (N, Ty/r, Cx)

                # Decoder RNN
                for j in range(len(self.decoder_rnn)):
                    y_f, dec_hidden[j] = self.decoder_rnn[j](y_, dec_hidden[j])  # (N, 1, Cx)
                    y_ = y_ + y_f  # (N, 1, Cx)

                # binary final frame prediction
                ff_hat = self.final_frame(y_)  # (N, Ty/r, 1)

                # Mel-spectrogram
                mel_hat = self.proj_mel(y_)  # (N, 1, n_mels*r)
                decoder_inputs = mel_hat[:, :, -args.n_mels * args.r:]  # last frame
                mels_hat.append(mel_hat)

                if (ff_hat[:, -1] > 0.5).sum() == len(ff_hat):
                    break

            mels_hat = torch.cat(mels_hat, dim=1)
            attention = torch.cat(attention, dim=1)

            # Decoder CBHG
            y_ = mels_hat.view(mels_hat.size(0), -1, args.n_mels)  # (N, Ty, n_mels)
            y_, _ = self.cbhg(y_)
            mags_hat = self.proj_mag(y_)

            return mels_hat, mags_hat, attention, ff_hat


class TPGST(nn.Module):
    """
    GST-Tacotron

    """

    def __init__(self):
        super(TPGST, self).__init__()
        self.name = 'TPGST'
        # len(args.vocab)
        self.embed = nn.Embedding(args.vocab_size, args.Ce, padding_idx=0)
        self.encoder = TextEncoder(hidden_dims=args.Cx)  # bidirectional
        self.GST = StyleTokenLayer(embed_size=args.Cx, n_units=args.Cx)
        self.tpnet = TPSENet(text_dims=args.Cx * 2, style_dims=args.Cx)
        self.decoder = AudioDecoder(enc_dim=args.Cx * 3, dec_dim=args.Cx)

    def forward(self, texts, prev_mels, refs=None, synth=False, ref_mode=True):
        """
        :param texts: (N, Tx) Tensor containing texts
        :param prev_mels: (N, Ty/r, n_mels*r) Tensor containing previous audio
        :param refs: (N, Ty, n_mels) Tensor containing reference audio
        :param synth: Boolean. whether it synthesizes.
        :param ref_mode: Boolean. whether it is reference mode

        Returns:
            :mels_hat: (N, Ty/r, n_mels*r) Tensor. mel spectrogram
            :mags_hat: (N, Ty, n_mags) Tensor. magnitude spectrogram
            :attentions: (N, Ty/r, Tx) Tensor. seq2seq attention
            :style_attentions: (N, n_tokens) Tensor. Style token layer attention
            :ff_hat: (N, Ty/r, 1) Tensor for binary final prediction
            :style_emb: (N, 1, E) Tensor. Style embedding
        """
        x = self.embed(texts)  # (N, Tx, Ce)
        text_emb, enc_hidden = self.encoder(x)  # (N, Tx, Cx*2)
        tp_style_emb = self.tpnet(text_emb)
        if synth:
            style_emb, style_attentions = tp_style_emb, None
        else:
            style_emb, style_attentions = self.GST(refs, ref_mode=ref_mode)  # (N, 1, E), (N, n_tokens)

        tiled_style_emb = style_emb.expand(-1, text_emb.size(1), -1)  # (N, Tx, E)
        memory = torch.cat([text_emb, tiled_style_emb], dim=-1)  # (N, Tx, Cx*2+E)
        mels_hat, mags_hat, attentions, ff_hat = self.decoder(prev_mels, memory, synth=synth)
        return mels_hat, mags_hat, attentions, style_attentions, ff_hat, style_emb, tp_style_emb



class Conv1d(nn.Conv1d):
    """
    :param in_channels: Scalar
    :param out_channels: Scalar
    :param kernel_size: Scalar
    :param activation_fn: activation function
    :param drop_rate: Scalar. dropout rate
    :param stride: Scalar
    :param padding: padding type
    :param dilation: Scalar
    :param groups: Scalar
    :param bias: Boolean.
    :param bn: Boolean. whether it uses batch normalization

    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                 stride=1, padding='same', dilation=1, groups=1, bias=True, bn=False):
        self.activation_fn = activation_fn
        self.drop_rate = drop_rate
        if padding == 'same':
            padding = kernel_size // 2 * dilation
            self.even_kernel = not bool(kernel_size % 2)
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)
        self.drop_out = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.001) if bn else None

    def forward(self, x):
        """
        :param x: (N, C_in, T) Tensor.

        Returns:
            y: (N, C_out, T) Tensor.

        """
        y = super(Conv1d, self).forward(x)
        y = self.batch_norm(y) if self.batch_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        y = self.drop_out(y) if self.drop_out is not None else y
        y = y[:, :, :-1] if self.even_kernel else y
        return y

class TPSENet(nn.Module):
    """
    Predict speakers from style embedding (N-way classifier)

    """

    def __init__(self, text_dims, style_dims):
        super(TPSENet, self).__init__()
        self.conv = nn.Sequential(
            Conv1d(text_dims, style_dims, 3, activation_fn=torch.relu, bn=True, bias=False),
            # mm.Conv1d(style_dims, style_dims, 3, activation_fn=torch.relu, bn=True, bias=False)
        )
        self.gru = nn.GRU(style_dims, style_dims, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(style_dims * 2, style_dims)
        # self.net = nn.Linear(args.Cx, args.n_speakers)

    def forward(self, text_embedding):
        """
        :param text_embedding: (N, Tx, E)

        Returns:
            :y_: (N, 1, n_speakers) Tensor.
        """
        te = text_embedding.transpose(1, 2)  # (N, E, Tx)
        h = self.conv(te)
        h = h.transpose(1, 2)  # (N, Tx, C)
        out, _ = self.gru(h)
        se = self.fc(out[:, -1:, :])
        se = torch.tanh(se)
        return se


class EndpointHandler():
    def __init__(
            self,
            load_model="latest",
            synth_mode="synthesize",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.device = device
        self.model = TPGST().to(device)

        if load_model.lower() == 'best':
            ckpt = pd.read_csv(os.path.join(args.logdir, self.model.name, 'ckpt.csv'), sep=',', header=None)
            ckpt.columns = ['models', 'loss']
            model_path = ckpt.sort_values(by='loss', ascending=True).models.loc[0]
            model_path = os.path.join(args.logdir, self.model.name, model_path)
        elif 'pth.tar' in load_model:
            model_path = load_model
        else:
            model_path = sorted(glob.glob(os.path.join(args.logdir, self.model.name, 'model-*.tar')))[-1]  # latest model
        state = torch.load(model_path, map_location=torch.device(device))
        self.model.load_state_dict(state['model'])
        args.global_step = state['global_step']
        print('The model is loaded. Step: {}'.format(args.global_step))
        self.model.eval()

    def __call__(self, data):
        """
        data args:
            inputs (:obj: `str` | `PIL.Image` | `np.array`)
            kwargs
        Return:
            A :obj:`list` | `dict`: will be serialized and returned
        """
        inputs = data.pop("inputs", data)

        with torch.no_grad():
            text = inputs.split(" ")
            text = torch.Tensor(np.array([_symbol_to_id[ch] for ch in text]))
            texts = text.unsqueeze(1)
            GO_frames = torch.zeros([texts.shape[0], 1, args.n_mels * args.r]).to(DEVICE)
            mels_hat, mags_hat, A, _, _, se, _ = self.model(texts, GO_frames, synth=True)
            mels_hat = mels_hat.cpu().numpy()
            return {
                "label": mels_hat[0],
                "score": 0
            }

if __name__ == "__main__":
    my_handler = EndpointHandler()

    holiday_payload = {"inputs": "Today is a though day", "date": "2022-07-04"}
    holiday_payload = my_handler(holiday_payload)
    print(holiday_payload)