import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):    
        x = self.embed(captions[:, : -1])
        x = torch.cat((features.unsqueeze(1), x), 1)
        hiddens, hs = self.lstm(x)
        outputs = self.linear(hiddens)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ids = []
        #inputs = inputs.squeeze()
        for i in range(20):
            #print(i)
            x, states = self.lstm(inputs, states)
            temp = self.linear(x.squeeze(1))
            pred = temp.max(1)[1]
            ids.append(pred.tolist()[0])
            inputs = self.embed(pred)
            inputs = inputs.unsqueeze(1)
        return ids
            