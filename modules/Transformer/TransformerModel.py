import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import Transformer_config as config
# Transformerのモデル
class TransformerModel(nn.Module):
    def __init__(
        self, input_dim, model_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(
            model_dim, config.max_len
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(
            model_dim, input_dim
        )
        self.classifier = nn.Linear(input_dim, num_classes)  # クラス分類のための出力層

    def forward(self, src, mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        # 入力の形状を変更 (B, L, D) -> (L, B, D)
        src = src.permute(1, 0, 2)
        if mask is not None: # マスクがある場合はそれを使用
            # マスクを反転（PyTorchのTransformerはパディング部分をTrueとして扱う）
            mask = ~mask
            output = self.transformer_encoder(src, src_key_padding_mask=mask)
        else:
            output = self.transformer_encoder(src)
        # グローバルプーリング（平均）を適用
        output = output.mean(dim=0)
        output = self.decoder(output)
        output = self.classifier(output)
        output = F.softmax(output, dim=1)
        #max_indices = torch.argmax(output, dim=1)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        position = torch.arange(max_len).unsqueeze(1).float() #max_lenに応じて配列を生成している。max_len = 10 [1.0, 2.0 ..., 10.0]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model) #0を埋めている。
        pe[:, 0::2] = torch.sin(position * div_term) #偶数の行を指定
        pe[:, 1::2] = torch.cos(position * div_term) #奇数の行を指定
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1), : x.size(2)] #位置エンコーディングを加えている。
        return self.dropout(x)

