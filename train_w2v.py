import config
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from utils import get_clear_tokens, get_stopwords, extract_qa_pairs


def data2txt():
    """
    调整为训练为w2v文件格式： 文本格式是 单词空格分开，一行为一个文档

    """
    pairs = extract_qa_pairs()
    stopwords = get_stopwords()
    with open(config.qa_txt, 'w') as f:
        for pair in pairs:
            f.write(' '.join(get_clear_tokens(pair[0], stopwords, word=True)) + '\n')
            f.write(' '.join(get_clear_tokens(pair[1], stopwords, word=True)) + '\n')


def train_w2v_model():
    """
    训练w2v模型
    """
    sentences = LineSentence(config.qa_txt)
    model = Word2Vec(sentences, sg=1, size=config.DIM_EMBEDDING, window=5, min_count=1, negative=3, iter=500)
    model.save('w2v.model')


if __name__ == '__main__':
    # data2txt()
    train_w2v_model()

    # model, w2v_dict = load_w2v()
    # print(model.wv['四川'])
    # for word in model.wv.index2word:
    #     print(word)
    #     break
    # print(model.wv.index2word[0])
    # print(get_w2i_dict())
    # print(model.wv.similarity('英国牛津大学', '英国剑桥大学'))
