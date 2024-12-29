import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def get_params(vocab_size, num_hiddens, device):
    # vocab_size 是不是就是num_steps?
    num_inputs = num_outputs = vocab_size

    # 本来是一个个的词，对其进行onehot编码， 生成的向量长度 是vocab_size, 这个好理解！！！
    # 输出是一个分类函数，类别就是vocab_size
    def normal(shape):
        """
        最简单的初始化函数

        :param shape:
        :return:
        """
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))  # 输入到隐层
    W_hh = normal((num_hiddens, num_hiddens))  # 隐层到隐层   比mlp多了这个东西！
    b_h = torch.zeros(num_hiddens, device=device)  # 隐藏层
    W_hq = normal((num_hiddens, num_outputs))  # 隐藏层到输出
    b_q = torch.zeros(num_outputs, device=device)  # 输出
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_run_state(batch_size, num_hiddens, device):
    # 初始化时返回隐藏状态
    # 在0时刻的时候，没有上一刻的隐藏状态
    # 大小是 批量大小 * num_hiddens
    # 这个到底是个啥? 是书上的Ht
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:  # 不同时间步
        # 先更新H
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)  # 将所有时间步的输入放在一起
    return torch.cat(outputs, dim=0), (H,)  # cat是为什么拼起来？2维矩阵，列数 vocabsize /行数 批量大小 * 时间长度  # 输入更新后的隐藏状态H


class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix 后面生成新字符"""
    # num_preds 预测的字符数
    state = net.begin_state(batch_size=1, device=device)  # 预测是对一句话进行预测，所以batch_size=1
    outputs = [vocab[prefix[0]]]  # 第一个单词index，明白了！
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 它接收一个名为outputs的变量，并返回一个用PyTorch张量表示的数据。
    # 这tmd的什么意思？获得最新时间步的xt？？然后传入net，xt会被onehot编码！
    for y in prefix[1:]:
        _, state = net(get_input(), state)  # 前一个时间位xt-1 和 ht-1来计算最新的 ht
        outputs.append(vocab[y])
    # 这一步是获得最新的state，"time traveller " 15个字符后的state
    for _ in range(num_preds):  # 开始进行预测
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]  # 将需要进行梯度计算的拿出来
    else:
        params = net.params
    # 取的所有层的params全都拿出来
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    # todo 这里有疑问？ 这里的p.grad??? 这里是什么意思，先做梯度再裁剪？？还是怎么的
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期（定义见第八章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # ??
    for X, Y in train_iter:
        if state is None or use_random_iter:  # use_random_iter 是随机采样还是顺序采样！
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()  # 跟计算图相关的去掉
            else:
                for s in state:
                    s.detach_()
            # 这里的逻辑不是很清楚，怎么搞的？
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()  # concat 就是多分类问题
        if isinstance(updater, torch.optim.Optimizer):  # Optimizer是父类
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()  # 困惑度


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel="perplexity", legend=["train"], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)   # lr 表示学习率
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict("time traveller "))
            animator.add(epoch + 1, [ppl])
    print(f"困惑度{ppl:.1f}, {speed:.1f}词元/秒(str(device)")
    print(predict("time traveller "))
    print(predict("traveller"))


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    # batch_size 每一批大小 ？
    # num_steps 样本词长度？
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # 迭代器学习一下？？
    print(F.one_hot(torch.tensor([0, 3]), len(vocab)))  # 这里len(vocab)==28是什么意思？
    # 给你一个向量为 【0， 2】， 然后给你一个长度比如是28，那么转换为
    # tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0]])
    # 为啥是第三个为0？？？ 测试下来是2的位置， 3的话再往后移动一位

    # 小批量数据形状是  批量大小，时间步数？ 32 和 35
    # todo这里做个例子， 批量为2和步长为5
    X = torch.arange(10).reshape((2, 5))
    print(F.one_hot(X.T, 28).shape)
    # tensor([[0, 5],
    #         [1, 6],
    #         [2, 7],
    #         [3, 8],
    #         [4, 9]])  为什么要这么做呢？把时间放在前面，转化为 [时间, 小大，特征] 这样从时间角度来看是连续的  【时间, 【小大，特征】】

    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_run_state, rnn)
    state = net.begin_state(X.shape[0], d2l.try_gpu())
    Y, new_state = net(X.to(d2l.try_gpu()), state)  # __call__
    Y.shape, len(new_state), new_state[0].shape
    print(Y.shape)
    print(len(new_state))
    print(new_state[0].shape)

    print(predict_ch8("time traveller ", 10, net, vocab, d2l.try_gpu()))

    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
    d2l.plt.show()
    pass
