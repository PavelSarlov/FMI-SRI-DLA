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
import random

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.t(torch.tensor(sents_padded, dtype=torch.long, device=device))
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName,device):
        self.load_state_dict(torch.load(fileName, map_location = device))

    def create_mask(self, src):
        return (src != self.padTokenIdx).permute(1, 0)
    
    def __init__(self, encoder, decoder, sourceWord2ind, targetWord2ind, startToken, unkToken, padToken, endToken):
        super(NMTmodel, self).__init__()
        self.sourceWord2ind = sourceWord2ind
        self.targetWord2ind = targetWord2ind
        self.startTokenIdx = sourceWord2ind[startToken]
        self.unkTokenIdx = sourceWord2ind[unkToken]
        self.padTokenIdx = sourceWord2ind[padToken]
        self.endTokenIdx = sourceWord2ind[endToken]
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio = 0.5):
        source_padded = self.preparePaddedBatch(source, self.sourceWord2ind)
        target_padded = self.preparePaddedBatch(target, self.targetWord2ind)
        source_lengths = [len(s) for s in source]

        target_len = target_padded.shape[0]
        batch_size = target_padded.shape[1]
        target_vocab_size = len(self.targetWord2ind)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(next(self.parameters()).device)

        encoder_outputs, hidden = self.encoder(source_padded, source_lengths)

        input = target_padded[0, :]
        mask = self.create_mask(source_padded)
        
        for t in range(1, target_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            input = target_padded[t] if teacher_force else output.argmax(1)

        Y_bar = target_padded[1:-1].flatten(0, 1) 
        Y_bar[Y_bar==self.endTokenIdx] = self.padTokenIdx 
        H = torch.nn.functional.cross_entropy(outputs[1:-1].flatten(0, 1), Y_bar, ignore_index=self.padTokenIdx)

        return H

    def translateSentence(self, sentence, targetIdToWord, limit=1000):
        device = next(self.parameters()).device
        tokens = [self.sourceWord2ind[w] if w in self.sourceWord2ind.keys() else self.unkTokenIdx for w in sentence]
        source = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(1)
        result = [self.startTokenIdx]
        Ht = set()

        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(source, [len(source)])

            mask = self.create_mask(source)

            for t in range(limit):
                target = torch.tensor([result[-1]], dtype=torch.long, device=device)

                output, hidden, _ = self.decoder(target, hidden, encoder_outputs, mask)

                # pred_token = output.argmax(1).item()

                # result.append(pred_token)

                # if pred_token == self.endTokenIdx:
                #     break

                topk = output.squeeze(0).topk(4).indices.tolist()
                Ht.add(topk)

                pred_token = next(token for token in topk if token != self.unkTokenIdx)
                result.append(pred_token)

                if pred_token == self.endTokenIdx:
                    break

        return [targetIdToWord[i] for i in result[1:]]

class GRUEncoder(torch.nn.Module):
    def __init__(self, embed_size, enc_hid_size, dec_hid_size, source_size, dropout):
        super(GRUEncoder, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.embed = torch.nn.Embedding(source_size, embed_size)
        self.projection = torch.nn.Linear(2 * enc_hid_size, dec_hid_size)
        self.gru = torch.nn.GRU(embed_size, enc_hid_size, bidirectional = True)

    def forward(self, source_padded, source_lengths):
        E = self.dropout(self.embed(source_padded))

        outputPacked, hidden = self.gru(torch.nn.utils.rnn.pack_padded_sequence(E, source_lengths, enforce_sorted=False))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(outputPacked)

        t = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = torch.nn.functional.relu(self.projection(t))

        return output, hidden
        

class GRUDecoder(torch.nn.Module):
    def __init__(self, embed_size, enc_hid_size, dec_hid_size, target_size, dropout):
        super(GRUDecoder, self).__init__()
        self.embed = torch.nn.Embedding(target_size, embed_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.attention = Attention(enc_hid_size, dec_hid_size)
        self.gru = torch.nn.GRU((enc_hid_size * 2) + embed_size, dec_hid_size)
        self.projection = torch.nn.Linear((enc_hid_size *  2) + dec_hid_size + embed_size, target_size)

    def forward(self, source, hidden, encoder_outputs, mask):
        source = source.unsqueeze(0)
        attn = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)

        E = self.dropout(self.embed(source))

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(attn, encoder_outputs).permute(1, 0, 2)

        t = torch.cat((E, weighted), dim = 2)
        output, hidden = self.gru(t, hidden.unsqueeze(0))

        E = E.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        Z = self.projection(torch.cat((output, weighted, E), dim = 1))
        return Z, hidden.squeeze(0), attn.squeeze(1)

class Attention(torch.nn.Module):
    def __init__(self, enc_hid_size, dec_hid_size):
        super(Attention, self).__init__()

        self.attn = torch.nn.Linear((enc_hid_size) * 2 + dec_hid_size, dec_hid_size)
        self.v = torch.nn.Linear(dec_hid_size, 1, bias = False)

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1,0,2)

        energy = torch.nn.functional.relu(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))

        attention = self.v(energy).squeeze(2).masked_fill(mask == 0, -1e10)

        return torch.nn.functional.softmax(attention, dim = 1)
