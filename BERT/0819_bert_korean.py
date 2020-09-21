import os
import codecs
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
import numpy as np
import pandas as pd
from konlpy.tag import Okt
import re
from tqdm import tqdm

SEQ_LEN = 128
BATCH_SIZE = 128
EPOCHS = 1
LR = 0.001
LOAD_DATA = False

pretrained_path = 'multi_cased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

word2idx = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        word2idx[token] = len(word2idx)
idx2word = {v: k for v, k in enumerate(word2idx)}

model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)
model.summary()

tokenizer = Tokenizer(word2idx)

movie_data = pd.read_csv('dataset/ratings.txt', header=0, delimiter='\t', quoting=3)
movie_data = movie_data.dropna()


def preprocessing(review, okt, remove_stopwords=False, stop_words=[]):
    review_text = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\s]", "", review)
    word_review = okt.morphs(review_text, stem=True)

    if remove_stopwords:
        word_review = [token for token in word_review if not token in stop_words]

    return word_review


stop_words = ['은', '는', '이', '가', '하', '아', '것', '들',
              '의', '있', '되', '수', '보', '주', '등', '한']
okt = Okt()
clean_review = []

for i, review in enumerate(movie_data['document']):
    # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
    if type(review) == str:
        p = preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words)
        clean_review.append(p)
    else:
        clean_review.append([])  # string이 아니면 비어있는 값 추가
        print(i, review)

    if i % 100 == 0:
        print('%d : %.2f%% 완료됨.' % (i, 100 * i / len(movie_data)))

clean_review_origin = []
for review in clean_review:
    clean_review_origin.append(review)

review_label = movie_data['label'].to_numpy()

def load_data(review_list):
    global tokenizer
    indices, sentiments = [], []
    for folder, sentiment in (('neg', 0), ('pos', 1)):
        folder = os.path.join(path, folder)
        for name in tqdm(os.listdir(folder)):
            with open(os.path.join(folder, name), 'r', encoding='UTF8') as reader:
                text = reader.read()
            ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
            indices.append(ids)
            sentiments.append(sentiment)
    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    mod = indices.shape[0] % BATCH_SIZE
    if mod > 0:
        indices, sentiments = indices[:-mod], sentiments[:-mod]
    return [indices, np.zeros_like(indices)], np.array(sentiments)

x, y = load_data(clean_review_origin)
