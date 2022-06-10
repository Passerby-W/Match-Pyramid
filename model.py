import json
import torch
import config
import torch.nn as nn
from logging import getLogger
import torch.nn.functional as F


logger = getLogger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MatchPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_len_q = config.MAX_SEQ_LEN
        self.max_len_a = config.MAX_SEQ_LEN
        self.conv1_size = config.conv1_size
        self.pool1_size = config.pool1_size
        self.conv2_size = config.conv2_size
        self.pool2_size = config.pool2_size
        self.dim_hidden = config.dim_hid

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.conv1_size[-1],
                               kernel_size=tuple(self.conv1_size[0:2]),
                               padding=0,
                               bias=True)
        nn.init.kaiming_normal_(self.conv1.weight)

        self.conv2 = nn.Conv2d(in_channels=self.conv1_size[-1],
                               out_channels=self.conv2_size[-1],
                               kernel_size=tuple(self.conv2_size[0:2]),
                               padding=0,
                               bias=True)

        self.pool1 = nn.AdaptiveMaxPool2d(tuple(self.pool1_size))
        self.pool2 = nn.AdaptiveMaxPool2d(tuple(self.pool2_size))
        self.linear1 = nn.Linear(self.pool2_size[0] * self.pool2_size[1] * self.conv2_size[-1], self.dim_hidden, bias=True)
        nn.init.kaiming_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(self.dim_hidden, config.dim_out, bias=True)
        nn.init.kaiming_normal_(self.linear2.weight)

        if logger:
            self.logger = logger
            self.logger.info("Hyper Parameters of MatchPyramid: %s" % json.dumps(
                {"Kernel": [self.conv1_size, self.conv2_size],
                 "Pooling": [self.pool1_size, self.pool2_size],
                 "MLP": self.dim_hidden}))

    def forward(self, q, a):
        # q,a:[batch, seq_len, dim]
        batch_size, seq_len_q, dim = q.size()
        seq_len_a = a.size()[1]
        pad_q = self.max_len_q - seq_len_q
        pad_a = self.max_len_a - seq_len_a

        # use cosine similarity since dim is too big for dot-product
        simi_img = torch.matmul(q, a.transpose(1, 2)) / torch.sqrt(torch.tensor(dim).to(device))
        if pad_q != 0 or pad_a != 0:
            simi_img = F.pad(simi_img, (0, pad_a, 0, pad_q))
        # assert simi_img.size() == (batch_size, self.max_len_q, self.max_len_a)
        simi_img = simi_img.unsqueeze(1)

        # [batch, 1, conv1_w, conv1_h]
        simi_img = F.relu(self.conv1(simi_img))
        # [batch, 1, pool1_w, pool1_h]
        simi_img = self.pool1(simi_img)
        # [batch, 1, conv2_w, conv2_h]
        simi_img = F.relu(self.conv2(simi_img))
        # # [batch, 1, pool2_w, pool2_h]
        simi_img = self.pool2(simi_img)
        # assert simi_img.size()[1] == 1
        # [batch, pool1_w * pool1_h * conv2_out]
        simi_img = simi_img.view(batch_size, -1)
        # output = self.linear1(simi_img)
        output = self.linear2(F.relu(self.linear1(simi_img)))
        return output


if __name__ == '__main__':
    q = torch.randn(6, 17, 100)
    a = torch.randn(6, 23, 100)
    m = MatchPyramid()
    c = m(q, a)
    print(c.shape)
