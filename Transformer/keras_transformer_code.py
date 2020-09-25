#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:09:36 2020

@author: jsjang
"""

from keras_transformer import get_model, decode
import pickle
import warnings
warnings.filterwarnings('ignore', 'tensorflow')



# 단어 목록 dict를 읽어온다.
with open('./dataset/vocabulary.pickle', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)
    
# 학습 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 읽어온다.
with open('./dataset/train_data.pickle', 'rb') as f:
    trainXE, trainXD, trainYD = pickle.load(f)
	
# 평가 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
with open('./dataset/eval_data.pickle', 'rb') as f:
    testXE, testXD, testYD = pickle.load(f)



###=========================================###
###=================model===================###
###=========================================###
model = get_model(
    token_num=max(len(word2idx), len(word2idx)),
    embed_dim=32,
    encoder_num=2,
    decoder_num=2,
    head_num=4,
    hidden_dim=128,
    dropout_rate=0.05,
    use_same_embed=False,  # Use different embeddings for different languages
)
model.compile('adam', 'sparse_categorical_crossentropy')
model.summary()


###=========================================###
###===========model load or fit=============###
###=========================================###
LOAD_MODEL = True
if LOAD_MODEL:
    MODEL_PATH = './dataset/transformer.h5'
    model.load_weights(MODEL_PATH)
    
else:
    model.fit(
    x=[trainXE, trainXD],
    y=trainYD,
    epochs=1,
    batch_size=32)
    model.save_weights(MODEL_PATH)






###====================================###
###===========predict 함수=============###
###===================================###
def ivec_to_word(q_idx):
    decoded = decode(
        model,
        q_idx,
        start_token=word2idx['<START>'],
        end_token=word2idx['<END>'],
        pad_token=word2idx['<PADDING>'],
    )
    decoded = ' '.join(map(lambda x: idx2word[x], decoded[1:-1]))
    return decoded





###===============================###
###===========chatbot=============###
###===============================###
MAX_SEQUENCE_LEN = 10
def chatting(n=100):
    for i in range(n):
        question = input('Q: ')
        
        if  question == 'quit':
            break
        
        q_idx = []
        for x in question.split(' '):
            if x in word2idx:
                q_idx.append(word2idx[x])
            else:
                q_idx.append(word2idx['<UNKNOWN>'])   # out-of-vocabulary (OOV)
        
        # <PADDING>을 삽입한다.
        if len(q_idx) < MAX_SEQUENCE_LEN:
            q_idx.extend([word2idx['<PADDING>']] * (MAX_SEQUENCE_LEN - len(q_idx)))
        else:
            q_idx = q_idx[0:MAX_SEQUENCE_LEN]
        
        answer = ivec_to_word(q_idx)
        print('A: ', answer)

chatting(100)
