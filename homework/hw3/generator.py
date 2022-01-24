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

    # X = torch.tensor([char2id[c] for c in result] if len(result) > 0 else [model.unkTokenIdx], dtype=torch.long)
    # E = model.embed(X[None, :])                                  

    with torch.no_grad():
        out = None
        hc_n = None
        X = None
        E = None

        if len(result) == 0:
            X = torch.tensor([model.unkTokenIdx], dtype=torch.long)
            E = model.embed(X[None, :])                                  
            out, hc_n = model.lstm(E)
        else:
            for c in result:
               X = torch.tensor([char2id[c]], dtype=torch.long) 
               E = model.embed(X[None, :])                                  
               out, hc_n = model.lstm(E, hc_n) if hc_n else model.lstm(E)

        while len(result) < limit:
            out = model.dropout(out[-1])
            Z = model.projection(out.flatten(0, 1))
            p = torch.nn.functional.softmax(Z / temperature, dim = 0)

            # На numpy не му се хареса много грешката в тензорните стойности
            # затова се наложи да използвам float64 и да нормализирам стойностите.
            # Не намерих по-подходящ начин.
            prob = np.array(p).astype(np.float64)
            prob /= prob.sum()

            c = np.random.choice(list(char2id.keys()), p=prob)
            if char2id[c] == model.endTokenIdx:
                break

            result += c

            X = torch.tensor([char2id[c]], dtype=torch.long)
            E = model.embed(X[None, :])
            out, hc_n = model.lstm(E, hc_n)
    
    #### Край на Вашия код
    #############################################################################

    return result
