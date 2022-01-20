#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch

def generateCode(model, char2id, startSentence, limit=1000, temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ
    
    result = startSentence[1:]

    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.
    
    model.eval()

    device = next(model.parameters()).device
    X = torch.tensor([char2id[c] for c in result] if len(result) > 0 else [model.unkTokenIdx], dtype=torch.long, device=device)
    E = model.embed(X[None, :])                                  

    with torch.no_grad():
        out, hc_n = model.lstm(E)

        while len(result) < limit:
            out = model.dropout(out[-1])
            Z = model.projection(out.flatten(0, 1))
            p = torch.nn.functional.softmax(Z, dim = 0)
            prob = np.array(p).astype(np.float64)
            prob /= prob.sum()

            c = np.random.choice(list(char2id.keys()), p=prob)
            if char2id[c] == model.endTokenIdx:
                break

            result += c

            X = torch.tensor([char2id[c]], dtype=torch.long, device=device)
            E = model.embed(X[None, :])

            out, hc_n = model.lstm(E, hc_n)
    
    #### Край на Вашия код
    #############################################################################

    return result
