#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))
    
    def __init__(self, parameter1, parameter2, parameter3, parameter4):
        super(NMTmodel, self).__init__()
        # self.encoder = encoder
        # self.decoder = decoder

    def forward(self, source, target):
        return H

    def translateSentence(self, sentence, limit=1000):
        self.eval()
        return result

class LSTMEncoderModel(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, word2ind, unkToken, padToken):
        super(EncoderModel, self).__init__()
        self.word2ind = word2ind
        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.lstm = torch.nn.LSTM(embed_size, hidden_size)
        self.embed = torch.nn.Embedding(len(word2ind), embed_size)
        self.projection = torch.nn.Linear(hidden_size,len(word2ind))

    def forward(self, ):

        

class LSTMDecoderModel(torch.nn.Module):
    def __init__(self, hidden_size, target_size):
        super(DecoderModel, self).__init__()
