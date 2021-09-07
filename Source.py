import nltk
import pandas as pd
import numpy as np
from collections import Counter
import random
import re
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
import time
def tokenize(text):
    text=re.sub(r"http\S+", "", text)
    a=nltk.TweetTokenizer(strip_handles=True,reduce_len=True).tokenize(text)
    a=[x for x in a if x != '.' and x != ',' and x !='#' and x!='?' and x!="'"and x!='"'and x!=':)' and x!=')'and x!='('and x!=':'and x!='/'
       and x!=';'and x!='['and x!=']'and x!='{'and x!='}'and x!='+'and x!='='and x!='-'and x!='_'and x!='*'
       and x!='&'and x!='^'and x!='%'and x!='$'and x!='@'and x!='!'and x!='`'and x!='~']
    return a

def get_representation(toate_cuvintele, how_many):
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd

def get_corpus_vocabulary(corpus):
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter

def text_to_bow(text, wd2idx):
    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features

def corpus_to_bow(corpus, wd2idx):
    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)
    return all_features

def write_prediction(out_file, predictions):
    with open(out_file, 'w') as fout:
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)

def cross_validate(k, data, labels):
    chunk_size=len(labels)//k
    indici=np.arange(0,len(labels))
    random.shuffle(indici)
    for i in range(0,len(labels),chunk_size):
        valid_indici=indici[i:i+chunk_size]
        train_indici=np.concatenate([indici[0:i],indici[i+chunk_size:]])
        train=data[train_indici]
        valid=data[valid_indici]
        y_train=labels[train_indici]
        y_valid=labels[valid_indici]
        yield train, valid, y_train, y_valid


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']
labels = train_df['label'].values
toate_cuvintele = get_corpus_vocabulary(corpus)
wd2idx, idx2wd = get_representation(toate_cuvintele, 3000)
data = corpus_to_bow(corpus, wd2idx)
test_data = corpus_to_bow(test_df['text'], wd2idx)

clf=MultinomialNB()
start=time.time()
clf.fit(data,labels)
end=time.time()
print('Time= ',end-start,'\n')
predictii1=clf.predict(test_data)
write_prediction('submision.csv', predictii1)
print(predictii1)


'''scoruri=[]
c=np.zeros((2,2))
for train,valid,y_train,y_valid in cross_validate(10,data,labels):
    clf.fit(train,y_train)
    predictii=clf.predict(valid)
    for true,pred in zip(y_valid,predictii):
        c[true,pred]+=1
    scor=f1_score(y_valid,predictii)
    scoruri.append(scor)
print(scoruri)
print('Scorul mediu este: ',np.mean(scoruri))
print('Matricea de confuzie este:\n',c)'''
