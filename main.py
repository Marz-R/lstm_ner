import torch
import pandas as pd
import numpy as np
from train import run_model

def prepare_data(link, word2index):
    # load data as dataframe
    dic = {'sentence':[], 'named_entity':[]}
    with open(link) as f:
        sentence = []
        labels = []
        for line in f:
            row = line.split(sep='\t')
            if len(row) == 4:
                sentence.append(row[0])
                labels.append(row[3].strip())
            elif line == "\n" and len(sentence)!=0: 
                dic['sentence'].append(sentence)
                dic["named_entity"].append(labels)
                sentence = []
                labels = []
    data = pd.DataFrame(dic)

    # word to index
    for idx in data.index:
        # make words lowercase
        data['sentence'][idx] = [x.lower() for x in data['sentence'][idx]]

        for word in data['sentence'][idx]:
            if word not in word2index.keys():
                word2index[word] = len(word2index)

    return data, word2index

def prepare_embeddings(link, word2index):
    # load pre-trained word embeddings from file
    embeddings_dict = {}
    with open(link, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    # create embedding matrix from word embeddings and data
    embedding_dim = 50
    embedding_matrix = torch.zeros((len(word2index)+1, embedding_dim))
    for word, index in word2index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = torch.from_numpy(embedding_vector)
    
    return embedding_matrix


# word & tag to index
tag2index = {"O": 0, "B-ORG": 1, "I-ORG": 2, "B-MISC": 3, "I-MISC": 4, "B-LOC": 5, "I-LOC": 6, "B-PER": 7, "I-PER": 8}
word2index = {}

train_data, word2index = prepare_data("data/train.conll", word2index)
dev_data, word2index = prepare_data("data/dev.conll", word2index)
test_data, word2index = prepare_data("data/test.conll", word2index)
embedding_matrix = prepare_embeddings("embeddings/glove.6B.50d.txt", word2index)

run_model(train_data, dev_data, test_data, word2index, tag2index, embedding_matrix, EPOCHS=1)
