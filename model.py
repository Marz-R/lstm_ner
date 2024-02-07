import torch
from torch import nn

# Define the class for tagger
class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, output_size, embeddings):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.word_embeddings.weight.data.copy_(embeddings)

        # bidirectional LSTM with input of glove word embeddings and output hidden states, hidden_dim = 100
        self.lstm = nn.LSTM(input_size=embeddings.shape[1], hidden_size=hidden_dim, bidirectional=True)

        # linear layer that maps hidden state to tags
        self.linear = nn.Linear(hidden_dim*2, output_size)

    def forward(self, inputs):
        embedded = self.word_embeddings(inputs)
        lstm_out, _ = self.lstm(embedded.view(len(inputs), 1, -1))
        linear_out = self.linear(lstm_out.view(len(inputs), -1))
        return linear_out

