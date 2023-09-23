import torch
from torch import nn
from d2l import torch as d2l


#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


#@save
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)   # bert特有的，2 要不是0要不是1
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))    # 增加一个位置编码，随机初始化，可学习

    def forward(self, tokens, segments, valid_lens):  # 多了一个segments
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        # 3个embedding加起来
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


#@save
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))  # 接一个有单隐层的mlp

    def forward(self, X, pred_positions):  # 将pred的位置的结果输出，然后传到mlp里面生成结果
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


#@save
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)  # 2分类问题

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)


#@save
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)

        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


if __name__ == '__main__':
    token_a = ["this", "movie", "is", "great"]
    token_b = ["i", "like", "it"]
    token_c = ["i", "dont", "like", "it"]
    print(get_tokens_and_segments(token_a, token_b))
    print(get_tokens_and_segments(token_c))
    """
    (['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>'], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    (['<cls>', 'i', 'dont', 'like', 'it', '<sep>'], [0, 0, 0, 0, 0, 0])
    """

    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                          ffn_num_hiddens, num_heads, num_layers, dropout)

    tokens = torch.randint(0, vocab_size, (2, 8))  # batch size = 8
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)
    print(encoded_X.shape)

    mlm = MaskLM(vocab_size, num_hiddens)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)
    print(mlm_Y_hat.shape)

    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
    print(mlm_l.shape)  # 2个batchsize 3个位置 =》6

    encoded_X = torch.flatten(encoded_X, start_dim=1)
    # NSP的输入形状:(batchsize，num_hiddens)
    nsp = NextSentencePred(encoded_X.shape[-1])
    nsp_Y_hat = nsp(encoded_X)
    # nsp_Y_hat.shape

    nsp_y = torch.tensor([0, 1])
    nsp_l = loss(nsp_Y_hat, nsp_y)
    # nsp_l.shape
    pass
