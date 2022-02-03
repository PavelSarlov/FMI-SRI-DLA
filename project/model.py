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
import torch.nn.functional as F
import random

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.tensor(sents_padded, dtype=torch.long, device=self.device)
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName,device):
        self.load_state_dict(torch.load(fileName, map_location = device))

    def make_src_mask(self, src):
        return (src != self.padTokenIdx).unsqueeze(1).unsqueeze(2)   # [batch size, 1, 1, src len]
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.padTokenIdx).unsqueeze(1).unsqueeze(2)  # [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()  # [trg len, trg len]
        return trg_pad_mask & trg_sub_mask  # [batch size, 1, trg len, trg len]

    def get_topk(self, k, candidates, weights, alpha):
        lenghts = torch.count_nonzero(weights, dim = 1)
        norm_sum = torch.sum(weights, dim = 1) / torch.pow(lenghts, alpha)
        topk = norm_sum.topk(k).indices
        return candidates[topk], weights[topk]
    
    def __init__(self, encoder, decoder, device, sourceWord2ind, targetWord2ind, startToken, unkToken, padToken, endToken):
        super(NMTmodel, self).__init__()
        self.device = device
        self.sourceWord2ind = sourceWord2ind
        self.targetWord2ind = targetWord2ind
        self.targetInd2Word = { v: k for k, v in targetWord2ind.items() }
        self.startTokenIdx = sourceWord2ind[startToken]
        self.unkTokenIdx = sourceWord2ind[unkToken]
        self.padTokenIdx = sourceWord2ind[padToken]
        self.endTokenIdx = sourceWord2ind[endToken]
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        src_padded = self.preparePaddedBatch(src, self.sourceWord2ind)    # [batch size, src len]
        trg_padded = self.preparePaddedBatch(trg, self.targetWord2ind)    # [batch size, trg len]
        
        src_mask = self.make_src_mask(src_padded)               # [batch size, 1, 1, src len]      
        trg_mask = self.make_trg_mask(trg_padded[:, :-1])       # [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src_padded, src_mask)                                           # [batch size, src len, hid dim]
        output, attention = self.decoder(trg_padded[:, :-1], enc_src, trg_mask, src_mask)      # [batch size, trg len, output dim], [batch size, n heads, trg len, src len]

        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg_padded = trg_padded[:,1:].contiguous().view(-1)

        H = torch.nn.functional.cross_entropy(output, trg_padded, ignore_index=self.padTokenIdx)

        return H

    def translateSentence(self, sentence, beam=False, limit=1000):
        if beam:
            return self.beamTranslate(sentence, limit)
        else:
            return self.greedyTranslate(sentence, limit)

    def greedyTranslate(self, sentence, targetIdToWord, limit=1000):
        tokens = [self.sourceWord2ind[w] if w in self.sourceWord2ind.keys() else self.unkTokenIdx for w in sentence]
        src = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        src_mask = self.make_src_mask(src)
        result = [self.startTokenIdx]

        with torch.no_grad():
            encoder_outputs = self.encoder(src, src_mask)

            for i in range(limit):
                trg = torch.tensor(result, dtype=torch.long, device=self.device).unsqueeze(0)

                trg_mask = self.make_trg_mask(trg)

                output, attn = self.decoder(trg, encoder_outputs, trg_mask, src_mask)
                output = output[:, -1, :].squeeze()

                sm = torch.nn.Softmax(0)
                output = sm(output)
                
                topk = output.topk(2).indices.tolist()

                pred_token = topk[0] if topk[0] != self.unkTokenIdx else topk[1]
                result.append(pred_token)

                if pred_token == self.endTokenIdx:
                    break

        return [self.targetInd2Word[i] for i in result[1:] if i != self.endTokenIdx]

    def beamTranslate(self, sentence, limit):
        beam_width = 2
        alpha = 0.7

        tokens = [self.sourceWord2ind[w] if w in self.sourceWord2ind.keys() else self.unkTokenIdx for w in sentence]
        src = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        src_mask = self.make_src_mask(src)

        candidates = torch.tensor([self.startTokenIdx], dtype=torch.long, device=self.device).repeat(beam_width, 1)
        weights = torch.tensor([0], device=self.device).repeat(beam_width, 1)

        with torch.no_grad():
            encoder_outputs = self.encoder(src, src_mask)

            for i in range(limit):
                finished = [j for j,c in enumerate(candidates) if c[-1].item() in [self.endTokenIdx, self.padTokenIdx]]
                cur_candidates = F.pad(candidates[finished], (0, 1), value=self.padTokenIdx) if finished else torch.empty(0, dtype=torch.long, device=self.device)
                cur_weights = F.pad(weights[finished], (0, 1)) if finished else torch.empty(0, device=self.device)

                cur_size = cur_candidates.shape[0]

                for j in range(beam_width):

                    if candidates[j, -1].item() in [self.endTokenIdx, self.padTokenIdx]:
                        continue

                    trg = candidates[j].unsqueeze(0)

                    trg_mask = self.make_trg_mask(trg)

                    output, attn = self.decoder(trg, encoder_outputs, trg_mask, src_mask)
                    output = output[:, -1, :].squeeze()

                    vocab_size = output.shape[0]

                    sm = torch.nn.LogSoftmax(0)
                    output = sm(output).unsqueeze(0).permute(1, 0).to(self.device)
                    indices = torch.arange(vocab_size).unsqueeze(0).permute(1, 0).to(self.device)

                    step_candidates = torch.cat((candidates[j].repeat(vocab_size, 1), indices), dim = 1)
                    step_weights = torch.cat((weights[j].repeat(vocab_size, 1), output), dim = 1)

                    cur_candidates = torch.cat((cur_candidates, step_candidates))
                    cur_weights = torch.cat((cur_weights, step_weights))
                    
                    if i == 0:
                        break

                if cur_size == len(cur_candidates):
                    break

                candidates, weights = self.get_topk(beam_width, cur_candidates, cur_weights, alpha)

        result = candidates[0].tolist()[1:]

        return [self.targetInd2Word[i] for i in result if i not in [self.endTokenIdx, self.padTokenIdx]]


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 1000):
        super().__init__()

        self.device = device
        
        self.tok_embedding = torch.nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = torch.nn.Embedding(max_length, hid_dim)
        
        self.layers = torch.nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        
        self.dropout = torch.nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)   # [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))  # [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask) # [batch size, src len, hid dim]
            
        return src

class EncoderLayer(torch.nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = torch.nn.LayerNorm(hid_dim)
        self.ff_layer_norm = torch.nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))  # [batch size, src len, hid dim]

        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))  # [batch size, src len, hid dim]
        
        return src

class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_k = torch.nn.Linear(hid_dim, hid_dim)
        self.fc_v = torch.nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = torch.nn.Linear(hid_dim, hid_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)  # [batch size, query len, hid dim]
        K = self.fc_k(key)    # [batch size, key len, hid dim]  
        V = self.fc_v(value)  # [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch size, n heads, key len, head dim]  
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)  # [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)  # [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()  # [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)  # [batch size, query len, hid dim]
        
        x = self.fc_o(x)  # [batch size, query len, hid dim]
        
        return x, attention

class PositionwiseFeedforwardLayer(torch.nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = torch.nn.Linear(hid_dim, pf_dim)
        self.fc_2 = torch.nn.Linear(pf_dim, hid_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))  # [batch size, seq len, pf dim]
        
        x = self.fc_2(x)  # [batch size, seq len, hid dim]
        
        return x

class Decoder(torch.nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 1000):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = torch.nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = torch.nn.Embedding(max_length, hid_dim)
        
        self.layers = torch.nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        
        self.fc_out = torch.nn.Linear(hid_dim, output_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)  # [batch size, trg len] 
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))  # [batch size, trg len, hid dim] 
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)  # [batch size, trg len, hid dim], [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)  # [batch size, trg len, output dim] 
            
        return output, attention

class DecoderLayer(torch.nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = torch.nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = torch.nn.LayerNorm(hid_dim)
        self.ff_layer_norm = torch.nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))  # [batch size, trg len, hid dim]

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))  # [batch size, trg len, hid dim]
        
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))  # [batch size, trg len, hid dim]
        
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention
