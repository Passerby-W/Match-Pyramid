import torch
from torch.utils.data import Dataset

import config


class MPDataset(Dataset):

    def __init__(self, data_dir='qa.txt'):
        self.data_list = list()
        with open(data_dir, 'r') as file:
            index = 0
            queries, answers, labels = [], [], []
            for line in file.readlines():
                labels.append(1)
                if index % 2 == 0:
                    query = line.strip().split(" ")
                    queries.append(query)
                else:
                    answer = line.strip().split(" ")
                    answers.append(answer)
                index += 1
            self.data_list = list(zip(queries, answers, labels))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def split_dataset(dataset):
    train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    return train_set, test_set


def collate_fn(data):
    queries, answers, labels = list(), list(), list()
    for data_item in data:
        queries.append(data_item[0])
        answers.append(data_item[1])
        labels.append(data_item[2])
    len_q = [len(sen) for sen in queries]
    len_a = [len(sen) for sen in answers]
    return queries, len_q, answers, len_a, labels


def collate_fn(data):
    sen1_list, sen2_list, label_list = list(), list(), list()
    for data_item in data:
        sen1_list.append(data_item[0])
        sen2_list.append(data_item[1])
        label_list.append(data_item[2])
    len1 = [len(sen) for sen in sen1_list]
    len2 = [len(sen) for sen in sen2_list]
    return sen1_list, len1, sen2_list, len2, label_list


def truncate(sen1, len1, sen2, len2, label, word2idx, max_seq_len=config.MAX_SEQ_LEN):
    def get_idx(w):
        if w in word2idx:
            return word2idx[w]
        else:
            return word2idx["UNK"]
    batch_size = len(sen1)
    max_len1 = min(max(len1), max_seq_len)
    max_len2 = min(max(len2), max_seq_len)
    len1 = torch.LongTensor(len1)
    len2 = torch.LongTensor(len2)
    label = torch.LongTensor(label)
    sen1_ts = torch.LongTensor(batch_size, max_len1).fill_(0)
    sen2_ts = torch.LongTensor(batch_size, max_len2).fill_(0)
    for i in range(batch_size):
        if len1[i] > max_seq_len:
            len1[i] = max_seq_len
        _sent1 = torch.LongTensor([get_idx(w) for w in sen1[i]])
        sen1_ts[i, :len1[i]] = _sent1[:len1[i]]
        if len2[i] > max_seq_len:
            len2[i] = max_seq_len
        _sent2 = torch.LongTensor([get_idx(w) for w in sen2[i]])
        sen2_ts[i, :len2[i]] = _sent2[:len2[i]]
    return sen1_ts, len1, sen2_ts, len2, label

if __name__ == '__main__':
    dataset = MPDataset()
    print(len(dataset))
    print(dataset[0])

