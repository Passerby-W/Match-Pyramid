# 1：import argparse
#
# 2：parser = argparse.ArgumentParser()
#
# 3：parser.add_argument()
#
# 4：parser.parse_args()

# 上面四个步骤解释如下：
# 首先导入该模块；
# 然后创建一个解析对象；
# 然后向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；
# 最后调用parse_args()方法进行解析；解析成功之后即可使用。
import os
import torch
import numpy as np
import argparse

from utils import init_logger, load_w2v
from src.model import MatchPyramidClassifier
from src.dataset import MSRPDataset

parser = argparse.ArgumentParser(
    description='Train and Evaluate MatchPyramid on MSRP dataset')

# main parameters
parser.add_argument("--data_path", type=str, default="./data/msr_paraphrase",
                    help="")
parser.add_argument("--dump_path", type=str, default="./dump/",
                    help="")
parser.add_argument("--embedding_path", type=str, default="data/glove.6B.300d.txt",
                    help="")
parser.add_argument("--max_seq_len", type=int, default=50,
                    help="")
parser.add_argument("--batch_size", type=int, default=32,
                    help="")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="")
parser.add_argument("--n_epochs", type=int, default=50,
                    help="")

# model parameters
parser.add_argument("--dim_embedding", type=int, default=300,
                    help="")
parser.add_argument("--dim_output", type=int, default=2,
                    help="")

parser.add_argument("--conv1_size", type=str, default="5_5_8",
                    help="")
parser.add_argument("--pool1_size", type=str, default="10_10",
                    help="")
parser.add_argument("--conv2_size", type=str, default="3_3_16",
                    help="")
parser.add_argument("--pool2_size", type=str, default="5_5",
                    help="")
parser.add_argument("--mp_hidden", type=int, default="128",
                    help="")
parser.add_argument("--dim_out", type=int, default="2",
                    help="")

# parse arguments
params = parser.parse_args()

# check parameters

logger = init_logger(params)

params.word2idx, params.glove_weight = load_w2v(params.embedding_path, params.dim_embedding)

train_data = MSRPDataset(params.data_path, data_type="train")
test_data = MSRPDataset(params.data_path, data_type="test")

params.train_data = train_data
params.test_data = test_data

mp_model = MatchPyramidClassifier(params)
mp_model.run()