import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassF1Score
from model import LSTMTagger

def make_tensor(seq, id_list):
    idx = [id_list[w] for w in seq]
    return torch.tensor(idx, dtype=torch.long)

def run_model(train_data, dev_data, test_data, word2index, tag2index, embedding_matrix, HIDDEN_DIM = 100, EPOCHS = 20):

    model = LSTMTagger(HIDDEN_DIM, len(tag2index), embedding_matrix) 

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    f1 = MulticlassF1Score(num_classes=len(tag2index), average='macro')

    # train and dev 
    for epoch in range(EPOCHS):
        print("Epoch ", epoch+1)
        train_loss = 0
        model.train() # train mode
        for id in train_data.index:
            model.zero_grad()

            inputs = make_tensor(train_data['sentence'][id], word2index)
            tags = make_tensor(train_data['named_entity'][id], tag2index)

            result = model(inputs)

            loss = loss_function(result, tags)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # to visualize training process
            print("\rTraining: %d/%d"%(id+1, len(train_data.index)), end="")
        print(" mean loss:", train_loss/len(train_data.index))

        dev_f1_sum = 0
        model.eval() # evaluation mode
        for id in dev_data.index:
            model.zero_grad()

            inputs = make_tensor(dev_data['sentence'][id], word2index)
            tags = make_tensor(dev_data['named_entity'][id], tag2index)

            result = model(inputs)
            loss = loss_function(result, tags)

            dev_f1 = f1(result, tags)
            dev_f1_sum += dev_f1

            # dev f1 score
            print("\rDev: %d/%d"%(id+1, len(dev_data.index)), end="")
        print(" Dev F1 score: ", dev_f1_sum/len(dev_data))

    # test
    with torch.no_grad():
        print("Final")
        test_f1_sum = 0
        model.eval() # evaluation mode
        for id in test_data.index:
            model.zero_grad()

            inputs = make_tensor(test_data['sentence'][id], word2index)
            tags = make_tensor(test_data['named_entity'][id], tag2index)

            result = model(inputs)

            test_f1 = f1(result, tags)
            test_f1_sum += test_f1

            print("\rTesting: %d/%d"%(id+1, len(test_data.index)), end="")
        print(" Test F1 score: ", test_f1_sum/len(test_data))

