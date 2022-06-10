data_path = "data_cn.json"
stopwords_path = 'cn_stopwords.txt'
qa_txt = 'qa.txt'

MAX_SEQ_LEN = 64
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
N_EPOCH = 10

DIM_EMBEDDING = 256
DIM_HIDDEN = 128
DIM_OUTPUT = 2


conv1_size = [7, 7, 8]
pool1_size = [10, 10]
conv2_size = [3, 3, 16]
pool2_size = [5, 5]

dim_hid = 300
dim_out = 2