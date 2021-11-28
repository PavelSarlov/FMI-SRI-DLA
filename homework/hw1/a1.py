#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2021/2022
#############################################################################

### Домашно задание 1
###
### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
### conda activate tii
###
### Ако все още нямате създадено обкръжение прочетете файла README.txt за инструкции


########################################
import numpy as np
import random

alphabet = ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ь', 'ю', 'я']

def extractDictionary(corpus):
    dictionary = set()
    for doc in corpus:
        for w in doc:
            if w not in dictionary: dictionary.add(w)
    return dictionary

def editDistance(s1, s2):
    #### функцията намира разстоянието на Левенщайн-Дамерау между два низа
    #### вход: низовете s1 и s2
    #### изход: минималният брой на елементарните операции ( вмъкване, изтриване, субституция и транспоциция на символи) необходими, за да се получи от единия низ другия

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-25 реда

    distMatrix = [[y+x for x in range(len(s2)+1)] for y in range(len(s1)+1)]
    
    for i in range(1,len(s1)+1):
        for j in range(1,len(s2)+1):
            cost = 0 if s1[i-1]==s2[j-1] else 1
            distMatrix[i][j] = min(distMatrix[i-1][j-1] + cost,
                                distMatrix[i][j-1] + 1,
                                distMatrix[i-1][j] + 1)
            if i>1 and j>1 and s1[i-1]==s2[j-2] and s1[i-2]==s2[j-1] and s1[i-1]!=s2[j-1]:
                distMatrix[i][j] = min(distMatrix[i][j],
                                distMatrix[i-2][j-2] + 1)
    return distMatrix[-1][-1]

    #### Край на Вашия код
    #############################################################################
    
def editOperations(s1, s2):
    #### функцията намира елементарни редакции, неободими за получаването на един низ от друг
    #### вход: низовете s1 и s2
    #### изход: списък с елементарните редакции ( идентитет, вмъкване, изтриване, субституция и транспоциция на символи) необходими, за да се получи втория низ от първия
    
    #### Например: editOperations('котка', 'октава') би следвало да връща списъка:
    ####    [('ко', 'ок'), ('т','т'), ('', 'а'), ('к', 'в'), ('а','а')]
    ####        |ко   |т |   |к  |а |
    ####        |ок   |т |а  |в  |а |
    ####        |Trans|Id|Ins|Sub|Id|
    ####
    #### Можете да преизползвате и модифицирате кода на функцията editDistance
    #############################################################################
    #### Начало на Вашия код.

    len1 = len(s1) + 1; len2 = len(s2) + 1
    distMatrix = [[0 for x in range(len2)] for y in range(len1)]
    for i in range(len2):
        distMatrix[0][i] = (i,0,max(i-1,0),'',s2[i-1] if i > 0 else '')
    for i in range(len1):
        distMatrix[i][0] = (i,max(i-1,0),0,s1[i-1] if i > 0 else '','')

    for i in range(1,len1):
        for j in range(1,len2):
            cost = 0 if s1[i-1]==s2[j-1] else 1
            distMatrix[i][j] = min((distMatrix[i-1][j-1][0] + cost, i-1, j-1, s1[i-1], s2[j-1]),
                                (distMatrix[i][j-1][0] + 1, i, j-1, '', s2[j-1]),
                                (distMatrix[i-1][j][0] + 1, i-1, j, s1[i-1], ''),
                                key = lambda x: x[0])
            if i>1 and j>1 and s1[i-1]==s2[j-2] and s1[i-2]==s2[j-1] and s1[i-1]!=s2[j-1]:
                distMatrix[i][j] = min(distMatrix[i][j],
                                (distMatrix[i-2][j-2][0] + 1, i-2, j-2, s1[i-2:i], s2[j-2:j]),
                                key = lambda x: x[0])
                
    result = []
    cell = (len(s1), len(s2))
    while any(cell):
        result += [distMatrix[cell[0]][cell[1]][3:]]
        cell = distMatrix[cell[0]][cell[1]][1:3]
    return result[::-1]

    #### Край на Вашия код
    #############################################################################

def computeOperationProbs(corrected_corpus,uncorrected_corpus,smoothing = 0.2):
    #### Функцията computeOperationProbs изчислява теглата на дадени елементарни операции (редакции)
    #### Теглото зависи от конкретните символи. Използвайки корпусите, извлечете статистика. Използвайте принципа за максимално правдоподобие. Използвайте изглаждане. 
    #### Вход: Корпус без грешки, Корпус с грешки, параметър за изглаждане. С цел простота може да се счете, че j-тата дума в i-тото изречение на корпуса с грешки е на разстояние не повече от 2 (по Левенщайн-Дамерау) от  j-тата дума в i-тото изречение на корпуса без грешки.
    #### Следва да се използват функциите generateCandidates, editOperations, 
    #### Помислете как ще изберете кандидат за поправка измежду всички възможни.
    #### Важно! При изтриване и вмъкване се предполага, че празния низ е представен с ''
    #### Изход: Речник, който по зададена наредена двойка от низове връща теглото за операцията.
    
    #### Първоначално ще трябва да преброите и запишете в речника operations броя на редакциите от всеки вид нужни, за да се поправи корпуса с грешки. След това изчислете съответните вероятности.
    
    operations = {} # Брой срещания за всяка елементарна операция + изглаждане
    operationsProb = {} # Емпирична вероятност за всяка елементарна операция
    for c in alphabet:
        operations[(c,'')] = smoothing    # deletions
        operations[('',c)] = smoothing    # insertions
        for s in alphabet:
            operations[(c,s)] = smoothing    # substitution and identity
            if c == s:    
                continue
            operations[(c+s,s+c)] = smoothing    # transposition

    #############################################################################
    #### Начало на Вашия код.

    for sent1,sent2 in zip(corrected_corpus, uncorrected_corpus):
        for word1,word2 in zip(sent1,sent2):
            for op in editOperations(word1, word2):
                if op in operations:
                    operations[op] += 1

    operationsCorpus = sum(int(o) for o in operations.values())
    for op,val in operations.items():
        operationsProb[op] = val/(smoothing*len(operations)+operationsCorpus)
    
    #### Край на Вашия код.
    #############################################################################
    return operationsProb

def operationWeight(a,b,operationProbs):
    #### Функцията operationWeight връща теглото на дадена елементарна операция
    #### Вход: Двата низа a,b, определящи операцията.
    ####       Речник с вероятностите на елементарните операции.
    #### Важно! При изтриване и вмъкване се предполага, че празния низ е представен с ''
    #### изход: Теглото за операцията
    
    if (a,b) in operationProbs.keys():
        return -np.log(operationProbs[(a,b)])
    else:
        print("Wrong parameters ({},{}) of operationWeight call encountered!".format(a,b))

def editWeight(s1, s2, operationProbs):
    #### функцията editWeight намира теглото между два низа
    #### За намиране на елеметарните тегла следва да се извиква функцията operationWeight
    #### вход: низовете s1 и s2 и речник с вероятностите на елементарните операции.
    #### изход: минималното тегло за подравняване, за да се получи втория низ от първия низ
    
    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-25 реда

    len1 = len(s1) + 1; len2 = len(s2) + 1
    distMatrix = [[0 for x in range(len2)] for y in range(len1)]
    for i in range(1, len2):
        distMatrix[0][i] += operationWeight('', s2[i-1], operationProbs) + distMatrix[0][i-1]
    for i in range(1, len1):
        distMatrix[i][0] += operationWeight(s1[i-1], '', operationProbs) + distMatrix[i-1][0]

    for i in range(1,len1):
        for j in range(1,len2):
            distMatrix[i][j] = min(distMatrix[i-1][j-1] + operationWeight(s1[i-1], s2[j-1], operationProbs),
                                distMatrix[i][j-1] + operationWeight('', s2[j-1], operationProbs),
                                distMatrix[i-1][j] + operationWeight(s1[i-1], '', operationProbs))
            if i>1 and j>1 and s1[i-1]==s2[j-2] and s1[i-2]==s2[j-1] and s1[i-1]!=s2[j-1]:
                distMatrix[i][j] = min(distMatrix[i][j],
                                distMatrix[i-2][j-2] + operationWeight(s1[i-2:i], s2[j-2:j], operationProbs))
    return distMatrix[-1][-1]

    #### Край на Вашия код. 
    #############################################################################

def generateEdits(q):
    ### помощната функция, generateEdits по зададена заявка генерира всички възможни редакции на разстояние едно от тази заявка.
    ### Вход: заявка като низ q
    ### Изход: Списък от низове на разстояние 1 по Левенщайн-Дамерау от заявката
    ###
    ### В тази функция вероятно ще трябва да използвате азбука, която е дефинирана с alphabet
    ###
    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-15 реда
    
    result = []
    for i in range(len(q)+1):
        result.append(q[:i] + q[i+1:]) # deletions
        result.append(q[:i] + q[i:i+2][::-1] + q[i+2:]) # transpositions
        for c in alphabet:
            result.append(q[:i] + c + q[i:]) # insertions
            if i<len(q) and c != q[i]:
                result.append(q[:i] + c + q[i+1:]) # substitutions
    result = set(result)
    result.remove(q)
    return list(result)
        
    #### Край на Вашия код
    #############################################################################


def generateCandidates(query,dictionary,operationProbs):
    ### Започва от заявката query и генерира всички низове НА РАЗСТОЯНИЕ <= 2, за да се получат кандидатите за корекция. Връщат се единствено кандидати, които са в речника dictionary.
        
    ### Вход:
    ###     Входен низ query
    ###     Речник с допустими (правилни) думи: dictionary
    ###     речник с вероятностите на елементарните операции.

    ### Изход:
    ###     Списък от двойки (candidate, candidate_edit_log_probability), където candidate е низ на кандидат, а candidate_edit_log_probability е логаритъм от вероятността за редакция -- минус теглото.
    
    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-15 реда

    allCandidates = [query]
    for i in range(2):
        for x in list(map(generateEdits, allCandidates)):
            allCandidates += x
    allCandidates = set(allCandidates)
    allCandidates.remove(query)
    
    result = list(map(lambda x: (x, editWeight(x, query, operationProbs)), 
                      filter(lambda x: x in dictionary, allCandidates)))
    
    return result

    #### Край на Вашия код
    #############################################################################

def correctSpelling(r, dictionary, operationProbs):
    ### Функцията поправя корпус съдържащ евентуално сгрешени думи
    ### Генераторът на кандидати връща и вероятността за редактиране.
    ###
    ### Вход:
    ###    заявка: r - корпус от думи
    ###    речник с правилни думи: dictionary,
    ###    речник с вероятностите на елементарните операции: operationProbs
    ###    Изход: поправен корпус

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 5-15 реда

    corrected = [[] for s in r]
    for i,s in enumerate(r):
        for j,w in enumerate(s):
            if any(map(lambda x: x not in alphabet, w)) or w in dictionary:
                corrected[i].append(w)
                continue
            candidates = generateCandidates(w, dictionary, operationProbs)
            corrected[i].append(min(candidates, key=lambda x: x[1])[0] if candidates else w)
    return corrected

    #### Край на Вашия код
    #############################################################################

