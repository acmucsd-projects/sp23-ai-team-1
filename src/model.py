import torch
from torch import nn
from transformers import AutoModel


class ClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_classes, seq_length):
        super().__init__()
        self.rnn = nn.RNN(768, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size*seq_length, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x

class BERTWithClassifierHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        # ClassifierHead is already defined.
        self.classifier = ClassifierHead(20, num_classes, 500)
        
    def forward(self, x):
        x = self.bert(**x)
        x = x.last_hidden_state
        x = self.classifier(x)
        return x
