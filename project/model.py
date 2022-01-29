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
    
    def __init__(self, encoder, decoder, sourceWord2ind, targetWord2ind, unkToken, padToken, endToken):
        super(NMTmodel, self).__init__()
        self.sourceWord2ind = sourceWord2ind
        self.targetWord2ind = targetWord2ind
        self.unkTokenIdx = sourceWord2ind[unkToken]
        self.padTokenIdx = sourceWord2ind[padToken]
        self.endTokenIdx = sourceWord2ind[endToken]
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target):
        source_padded = self.preparePaddedBatch(source, self.sourceWord2ind)
        target_padded = self.preparePaddedBatch(target, self.targetWord2ind)
        source_lengths = [len(s) for s in source]

        target_len = target_padded.shape[0]
        batch_size = target_padded.shape[1]
        target_vocab_size = len(self.targetWord2ind)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(next(self.parameters()).device)

        _, hc_n = self.encoder(source_padded, source_lengths)

        input = target_padded[0, :]
        
        for t in range(1, target_len):
            output, hc_n = self.decoder(input, hc_n)

            outputs[t] = output

            input = output.argmax(1)

        Y_bar = target_padded[1:].flatten(0, 1) 
        H = torch.nn.functional.cross_entropy(outputs[1:].flatten(0, 1), Y_bar, ignore_index=self.padTokenIdx)

        return H

    def translateSentence(self, sentence, limit=1000):
        self.eval()
        return result

class LSTMEncoder(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, source_size, layers, dropout):
        super(LSTMEncoder, self).__init__()
        # self.lstm = torch.nn.LSTM(embed_size, hidden_size, layers = layers, dropout = dropout, bidirectional = True)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, layers, dropout = dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.embed = torch.nn.Embedding(source_size, embed_size)
        # self.projection = torch.nn.Linear(2 * hidden_size, source_size)
        # self.projection = torch.nn.Linear(hidden_size, source_size)

    def forward(self, source_padded, source_lengths):
        E = self.dropout(self.embed(source_padded))

        outputPacked, hc_n = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, enforce_sorted=False))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)

        return output, hc_n
        

class LSTMDecoder(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, target_size, layers, dropout):
        super(LSTMDecoder, self).__init__()
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, layers, dropout = dropout)
        self.embed = torch.nn.Embedding(target_size, embed_size)
        self.projection = torch.nn.Linear(hidden_size, target_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, source, hc_n):
        source = source.unsqueeze(0)

        E = self.dropout(self.embed(source))

        output, hc_n = self.lstm(E, hc_n)

        Z = self.projection(output.flatten(0,1))
        return Z, hc_n
