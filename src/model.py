# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # 注册为buffer（不参与训练）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入序列 [seq_len, batch_size, d_model]
        Returns:
            添加位置编码后的序列 [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: 查询矩阵 [batch_size, n_heads, seq_len_q, d_k]
            K: 键矩阵 [batch_size, n_heads, seq_len_k, d_k]
            V: 值矩阵 [batch_size, n_heads, seq_len_v, d_v]
            mask: 注意力掩码
        Returns:
            注意力输出和注意力权重
        """
        d_k = Q.size(-1)

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到V
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: 输入序列 [batch_size, seq_len, d_model]
            mask: 注意力掩码
        Returns:
            多头注意力输出 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = Q.size(0), Q.size(1)

        # 残差连接
        residual = Q

        # 线性投影并分头
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 如果有掩码，扩展到所有头
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = mask.repeat(1, self.n_heads, 1, 1)  # 扩展到所有头

        # 缩放点积注意力
        context, attn_weights = self.attention(Q, K, V, mask=mask)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        # 输出投影
        output = self.W_O(context)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: 输入 [batch_size, seq_len, d_model]
        Returns:
            前馈网络输出 [batch_size, seq_len, d_model]
        """
        residual = x

        # 两层前馈网络 + ReLU激活
        output = self.linear1(x)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output


class EncoderLayer(nn.Module):
    """编码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, src_mask):
        """
        Args:
            x: 编码器输入 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
        Returns:
            编码器输出 [batch_size, src_len, d_model]
        """
        # 自注意力子层
        x, attn_weights = self.self_attention(x, x, x, mask=src_mask)

        # 前馈网络子层
        x = self.feed_forward(x)

        return x, attn_weights


class DecoderLayer(nn.Module):
    """解码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Args:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            解码器输出 [batch_size, tgt_len, d_model]
        """
        # 自注意力子层（带未来掩码）
        x, self_attn_weights = self.self_attention(x, x, x, mask=tgt_mask)

        # 交叉注意力子层
        x, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, mask=src_mask)

        # 前馈网络子层
        x = self.feed_forward(x)

        return x, self_attn_weights, cross_attn_weights


class Encoder(nn.Module):
    """编码器"""

    def __init__(self, src_vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_length, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        # 编码器层堆叠
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, src_ids, src_mask):
        """
        Args:
            src_ids: 源序列索引 [batch_size, src_len]
            src_mask: 源序列掩码 [batch_size, 1, src_len]
        Returns:
            编码器输出 [batch_size, src_len, d_model]
        """
        # 词嵌入
        x = self.src_embedding(src_ids) * math.sqrt(self.d_model)

        # 位置编码
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        # 通过编码器层
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            attention_weights.append(attn_weights)

        return x, attention_weights


class Decoder(nn.Module):
    """解码器"""

    def __init__(self, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_length, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, tgt_ids, encoder_output, src_mask, tgt_mask):
        """
        Args:
            tgt_ids: 目标序列索引 [batch_size, tgt_len]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码 [batch_size, 1, src_len]
            tgt_mask: 目标序列掩码 [batch_size, tgt_len, tgt_len]
        Returns:
            解码器输出 [batch_size, tgt_len, d_model]
        """
        # 词嵌入
        x = self.tgt_embedding(tgt_ids) * math.sqrt(self.d_model)

        # 位置编码
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        # 通过解码器层
        self_attention_weights = []
        cross_attention_weights = []

        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)

        return x, self_attention_weights, cross_attention_weights


class Transformer(nn.Module):
    """完整的Transformer模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6,
                 n_heads=8, d_ff=2048, max_seq_length=100, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads,
                               d_ff, max_seq_length, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads,
                               d_ff, max_seq_length, dropout)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        # 参数初始化
        self._init_parameters()

    def _init_parameters(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_src_mask(self, src_ids, pad_idx=0):
        """创建源序列掩码"""
        # src_mask: [batch_size, 1, src_len]
        src_mask = (src_ids != pad_idx).unsqueeze(1)
        return src_mask

    def create_tgt_mask(self, tgt_ids, pad_idx=0):
        """创建目标序列掩码（包含未来掩码）"""
        batch_size, tgt_len = tgt_ids.size()

        # 填充掩码
        tgt_pad_mask = (tgt_ids != pad_idx).unsqueeze(1)  # [batch_size, 1, tgt_len]

        # 未来掩码（防止看到未来信息）
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
        tgt_sub_mask = tgt_sub_mask.unsqueeze(0)  # [1, tgt_len, tgt_len]

        # 组合掩码
        tgt_mask = tgt_pad_mask & tgt_sub_mask.to(tgt_ids.device)

        return tgt_mask

    def forward(self, src_ids, tgt_ids, src_pad_idx=0, tgt_pad_idx=0):
        """
        Args:
            src_ids: 源序列索引 [batch_size, src_len]
            tgt_ids: 目标序列索引 [batch_size, tgt_len]
        Returns:
            模型输出 [batch_size, tgt_len, tgt_vocab_size]
        """
        # 创建掩码
        src_mask = self.create_src_mask(src_ids, src_pad_idx)
        tgt_mask = self.create_tgt_mask(tgt_ids, tgt_pad_idx)

        # 编码器前向传播
        encoder_output, enc_attention_weights = self.encoder(src_ids, src_mask)

        # 解码器前向传播
        decoder_output, dec_self_attention_weights, dec_cross_attention_weights = \
            self.decoder(tgt_ids, encoder_output, src_mask, tgt_mask)

        # 输出投影
        output = self.output_layer(decoder_output)

        return {
            'output': output,
            'enc_attention': enc_attention_weights,
            'dec_self_attention': dec_self_attention_weights,
            'dec_cross_attention': dec_cross_attention_weights
        }

    def generate(self, src_ids, max_length=50, bos_idx=2, eos_idx=3, pad_idx=0):
        """生成翻译结果（贪婪解码）"""
        self.eval()

        batch_size = src_ids.size(0)
        device = src_ids.device

        # 编码源序列
        src_mask = self.create_src_mask(src_ids, pad_idx)
        encoder_output, _ = self.encoder(src_ids, src_mask)

        # 初始化目标序列（开始标记）
        tgt_ids = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            tgt_mask = self.create_tgt_mask(tgt_ids, pad_idx)

            # 解码器前向传播
            decoder_output, _, _ = self.decoder(
                tgt_ids, encoder_output, src_mask, tgt_mask)

            # 获取下一个token
            next_output = self.output_layer(decoder_output[:, -1, :])
            next_token = next_output.argmax(-1).unsqueeze(1)

            # 添加到目标序列
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

            # 如果所有序列都生成了结束标记，则停止
            if (next_token == eos_idx).all():
                break

        return tgt_ids


def create_transformer_model(src_vocab_size, tgt_vocab_size, config=None):
    """创建Transformer模型"""
    if config is None:
        config = {
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 4,
            'd_ff': 512,
            'max_seq_length': 100,
            'dropout': 0.1
        }

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        max_seq_length=config['max_seq_length'],
        dropout=config['dropout']
    )

    return model


# 测试代码
if __name__ == "__main__":
    # 测试模型
    batch_size, seq_len = 4, 20
    src_vocab_size, tgt_vocab_size = 10000, 15000

    # 创建模型
    model = create_transformer_model(src_vocab_size, tgt_vocab_size)

    # 创建模拟输入
    src_ids = torch.randint(0, src_vocab_size, (batch_size, seq_len))
    tgt_ids = torch.randint(0, tgt_vocab_size, (batch_size, seq_len))

    print(f"输入形状: src_ids {src_ids.shape}, tgt_ids {tgt_ids.shape}")

    # 前向传播
    with torch.no_grad():
        output = model(src_ids, tgt_ids)

    print(f"输出形状: {output['output'].shape}")
    print(f"编码器注意力权重数量: {len(output['enc_attention'])}")
    print(f"解码器自注意力权重数量: {len(output['dec_self_attention'])}")
    print(f"解码器交叉注意力权重数量: {len(output['dec_cross_attention'])}")

    # 测试生成
    generated = model.generate(src_ids, max_length=30)
    print(f"生成序列形状: {generated.shape}")

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")