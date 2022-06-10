import json
import gensim
import jieba
import torch
import numpy as np
from torch import nn

import config


def extract_qa(data_path=config.data_path):
    """读取qa pair，data size = 3024

    Args:
        data_path: default = 'config.data_path')    str

    Returns:
        pairs: list of query and answer pairs   list

    """
    query = []
    answer = []
    with open(data_path, 'r', encoding='utf-8') as f:
        qa = json.load(f)
    for pair in qa:
        query.append(qa[pair].get('question'))
        answer.append(qa[pair]['evidences'][pair + '#00'].get('evidence'))
    return query, answer


def extract_qa_pairs(data_path=config.data_path):
    """读取qa pair，data size = 3024

    Args:
        data_path: default = 'config.data_path'    str

    Returns:
        pairs: list of query and answer pairs   list

    """
    query = []
    answer = []
    with open(data_path, 'r', encoding='utf-8') as f:
        qa = json.load(f)
    for pair in qa:
        query.append(qa[pair].get('question'))
        answer.append(qa[pair]['evidences'][pair + '#00'].get('evidence'))
    pairs = list(zip(query, answer))
    return pairs


def is_contains_chinese(strs):
    """
    检验是否含有中文字符
    Args:
        strs:

    Returns:

    """
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
        if '\u0030' <= _char <= '\u0039':
            return True
        if ('\u0041' <= _char <= '\u005a') or ('\u0061' <= _char <= '\u007a'):
            return True
    return False


def tokenizer(text, word=True):
    """分词

    Args:
        text: str
        word: 词粒度 默认为False

    Returns:
        分词后的text

    """
    if word:
        tokens = list(jieba.cut(text))
        tokens = [token for token in tokens if is_contains_chinese(token)]
    else:
        tokens = [char for char in text if is_contains_chinese(char)]
    return tokens


def remove_stopwords(tokens, stopwords):
    """

    Args:
        tokens: list
        stopwords: list

    Returns:
        去掉停用词的token列表

    """
    return [token for token in tokens if token not in stopwords]


def get_clear_tokens(text, stopwords, word=True):
    """

    Args:
        text: str
        stopwords: 停用词字典
        word: 词粒度 默认为False

    Returns:
        clear_tokens: 去掉停用词的token列表

    """
    tokens = tokenizer(text, word=word)
    clear_tokens = remove_stopwords(tokens, stopwords)
    return clear_tokens


def get_stopwords(stopwords=config.stopwords_path):
    """

    Args:
        stopwords: 停用词文件路径

    Returns:
        stop_dict: 停用词字典

    """
    with open(stopwords, 'r') as f:
        stop_dict = {}
        words = f.readlines()
        for word in words:
            word = word.strip()
            stop_dict[word] = True
        return stop_dict


def load_w2v():
    """

    load w2v model

    """
    model = gensim.models.Word2Vec.load('w2v.model')
    w2i_dict = {}
    i2w_dict = {}
    w2i_dict['unk'] = 0
    i2w_dict[0] = 'unk'
    for index, word in enumerate(model.wv.index2word):
        w2i_dict[word] = index+1
        i2w_dict[index+1] = word
    return model, w2i_dict, i2w_dict


def creat_embedding(model, i2w_dict):  # 注意要先确定好训练语料里面有的单词
    vectors = torch.zeros([len(i2w_dict), config.DIM_EMBEDDING])
    for i in range(1, len(i2w_dict)):
        word = i2w_dict[i]
        vector = np.copy(model.wv[word])
        vectors[i, :] = torch.from_numpy(vector)
    embedding = nn.Embedding.from_pretrained(vectors)
    embedding.weight.requires_grad = True
    return embedding


if __name__ == '__main__':
    model, w2i_dict, i2w_dict = load_w2v()
    # print(model.wv.similarity('重庆', '山城'))
    # print(model.wv.most_similar('李商隐'))
    # print(i2w_dict)
    # print(extract_qa_pairs())
    # a,b = get_w2i_dict()
    a = creat_embedding(model, i2w_dict)
    print(a)
    # print(type(model.wv['四川']))
    # sw = get_stopwords()
    # print(get_clear_tokens('核电占发电量比例最大的是哪个国家?', sw))
    # print(get_clear_tokens('在世界主要工业大国中，法国核电的比例最高，核电占国家总发电量的78%，位居世界第二，日本的核电比例为', sw))
    # queries, answers = extract_qa()
    # maxLen = 0
    # minLen = len(queries[0])
    # count = 0
    # countLen = 0
    # for query in queries:
    #     count += 1
    #     countLen += len(query)
    #     if len(query) > maxLen:
    #         maxLen = len(query)
    #     if len(query) < minLen:
    #         minLen = len(query)
    # print(maxLen, minLen, countLen//count)
