import json
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from dataset import collate_fn, truncate

import config
from dataset import MPDataset, split_dataset
from model import MatchPyramid
from utils import creat_embedding, load_w2v

logger = getLogger()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MatchPyramidClassifier(object):

    def __init__(self):
        logger.info("Initializing MatchPyramidClassifier")
        dataset = MPDataset()
        train_data, test_data = split_dataset(dataset)
        self.train_data = train_data
        self.test_data = test_data
        self.current_epoch = 0

        self.model, self.w2i_dict, self.i2w_dict = load_w2v()
        self.embedding = creat_embedding(self.model, self.i2w_dict)

        self.matchPyramid = MatchPyramid()

        self.lr = config.LEARNING_RATE
        self.optimizer = torch.optim.Adam(
            list(self.embedding.parameters()) + list(self.matchPyramid.parameters()),
            lr=self.lr
        )
        self.embedding.to(device)
        self.matchPyramid.to(device)

    def run(self):
        for i in range(config.N_EPOCH):
            l = self.train()
            print(f'train loss: {l}')
            self.evaluate()
            self.current_epoch += 1

    def train(self):
        logger.info("Training in epoch %i" % self.current_epoch)
        self.embedding.train()
        self.matchPyramid.train()
        data_loader = DataLoader(self.train_data,
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=True,
                                 collate_fn=collate_fn)
        for data_iter in data_loader:
            queries, len_q, answers, len_a, labels = data_iter
            sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts = truncate(queries,
                                                                    len_q,
                                                                    answers,
                                                                    len_a,
                                                                    labels,
                                                                    self.w2i_dict)
            sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts = sen1_ts.to(device), \
                                                           len1_ts.to(device), \
                                                           sen2_ts.to(device), \
                                                           len2_ts.to(device), \
                                                           label_ts.to(device)
            sen1_embedding = self.embedding(sen1_ts)
            sen2_embedding = self.embedding(sen2_ts)
            mp_output = self.matchPyramid(sen1_embedding, sen2_embedding)
            loss = F.cross_entropy(mp_output, label_ts)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        logger.info("Evaluating in epoch %i" % self.current_epoch)
        self.embedding.eval()
        self.matchPyramid.eval()
        data_loader = DataLoader(self.test_data,
                                 batch_size=config.BATCH_SIZE,
                                 shuffle=False,
                                 collate_fn=collate_fn)
        pred_list = list()
        label_list = list()
        with torch.no_grad():
            for data_iter in data_loader:
                queries, len_q, answers, len_a, labels = data_iter
                sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts = truncate(queries,
                                                                        len_q,
                                                                        answers,
                                                                        len_a,
                                                                        labels,
                                                                        self.w2i_dict)
                sen1_ts, len1_ts, sen2_ts, len2_ts, label_ts = sen1_ts.to(device), \
                                                               len1_ts.to(device), \
                                                               sen2_ts.to(device), \
                                                               len2_ts.to(device), \
                                                               label_ts.to(device)
                sen1_embedding = self.embedding(sen1_ts)
                sen2_embedding = self.embedding(sen2_ts)
                mp_output = self.matchPyramid(sen1_embedding, sen2_embedding)
                predictions = mp_output.data.max(1)[1]
                pred_list.extend(predictions.tolist())
                label_list.extend(label_ts.tolist())
        acc = accuracy_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list)
        print("ACC score in epoch %i :%.4f" % (self.current_epoch, acc))
        print("F1 score in epoch %i :%.4f" % (self.current_epoch, f1))


if __name__ == '__main__':
    mp_model = MatchPyramidClassifier()
    mp_model.run()
