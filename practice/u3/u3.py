#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2021/2022
#############################################################################

### Упражнение 3
###
### За да работи програмата трябва корпуса от публицистични текстове за Югоизточна Европа,
### да се намира разархивиран в директорията, в която е програмата (виж упражнение 2).

### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
### conda activate tii

text1 = '''
    Ние виждаме една серия от управленски провали. Укрепено ли е правителството - не, то е несъществуващо. Правителството е несъществуващото - в този тежък момент, когато достигаме близо 1000 заразени на ден, ако говорим за здравния проблем.
    
    Протестът на хората е супер основателен, защото те виждат провала на властта в двете важни сфери - икономиката и здравето.
    
    Това заяви евродепутатът от ДСБ/ЕНП Радан Кънев пред бТВ.
    
    По думите му на 15 октомври е трябвало България да представи пред Европейската комисия проект за национален възстановителен план, което "ще определени икономическата съдба на поколения българи - това са парите за преодоляване на COVID кризата, но и парите за адаптация на Зелената сделка и парите, с които икономиката ни да компенсира минимум едно 10-годишно изоставане от европейските икономически политики".
    
    Преди да бъде представен пред Комисията, нали трябва да бъде представен пред обществото, пред бизнеса, синдикатите. Това е проект, който ще определи дали ние ще позволим да бъдем последни или ще си повярваме, че можем да бъдем богата държава, богато общество и че можем да имаме модерна икономика, добави той.
    
    Според него няма дебат с европейските партньори, с политици и с частния финансов сектор.
    '''

text2 = ''' Най-малко 34 души от афганистанските сили за сигурност, сред които и висш полицейски служител, са убити при нападение на талибаните в афганистанската провинция Тахар, съобщиха властите, цитирани от ДПА.
    
    Според полицията други 8 са били убити през нощта в друга част на страната, с което жертвите стават 42.
    
    Това е второто нападение на бунтовниците за последния месец, въпреи започналите в Катар миналия месец мирни преговори между правителството и талибаните. Припомняме, че за да стартират разговорите правителството освободи всички близо 5000 задържани талибани. След като условието беше изпълнено, на 12 септември преговорите за прекратяването на почти 20-годишната война започнаха.'''

text3 = '''
    ДОМ НА КИНОТО
    
    18:00 >> След огромния успех снощи при премиерата на дигитално възстановената първа серия на „МЕРА СПОРЕДЪ МЕРА“ в присъствието на режисьора Георги Дюлгеров, актьорите Руси Чанев и Стефан Мавродиев, оператора Радослав Спасов, художника Георги Тодоров-Жози и други членове на екипа, ви представяме втората част на филма - с напълно реставрирани картина и звук. Легендарната творба е създадена през 1981 година по сценарий на Руси Чанев и Георги Дюлгеров, по романа
    на Свобода Бъчварова „Литургия за Илинден“, режисирана от Дюлгеров и е едно от най-мащабните, епични и значими произведения на българското кино. В него присъстват както реални исторически личности, така и персонажи, родени от въображението на авторите с техните човешки драми, тревоги, съмнения.
    
    Разказът е обединен от метаморфозата на главния герой Дилбер Танас – от първичен овчар, част от патриархалната задруга до личност с индивидуално съзнание. Катализатор на тази промяна са борбите за независимост на македонските българи от 1901 до 1912 година. Съдбата на Дилбер Танас е метафора на общата ни история, в която човешки драми и политически игри променят изначалната идея на борбата и нейното значение. Това е първият български филм, пресъздаващ историята на Илинденско-Преображенското въстание през 1903 година, с участието на реални исторически личности - Апостол войвода, Христо Чернопеев, Яне Сандански, Пейо Яворов, Гоце Делчев, Георги Мучитан.
    
    Реставрираната визия, реализирана в Доли медия студио, е резултат от сканирането на целия негатив и неговото цялостно почистване; след това той е разчетен, направени са нови цветови корекции и е коригирана експозицията на кадрите, като са добавени визуални ефекти за подсилване на картината и подобряване на качеството й. Филмът е с изцяло нов звук – верен на оригинала, но преформатиран от моно в стерео 5+1, с добавени звукови ефекти. За да бъде улеснено възприемането на историята от младите поколения, при запазена оригинална фонограма със специфичните наречия и архаичната лексика, са добавени субтитри на литературен български език. '''

import nltk
from nltk.corpus import PlaintextCorpusReader
import sys

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    
    def start(self, count):
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    
    def tick(self, item):
        if item>0 and item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()

    def stop(self):
        sys.stdout.write("]\n")

corpus_root = '../JOURNALISM.BG/C-MassMedia'
myCorpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
fileNames = myCorpus.fileids()

classesSet = set( [ file[:file.find('/')] for file in fileNames ] )
classes = sorted(list(classesSet - {'Z','D-Society'}))

print(classes)

fullClassCorpus = [ [ myCorpus.words(file) for file in fileNames if file.find(c+'/')==0 ] for c in classes ]

import random

def splitClassCorpus(fullClassCorpus, testFraction = 0.1):
    testClassCorpus = []
    trainClassCorpus = []
    random.seed(42)
    for c in range(len(fullClassCorpus)):
        classList = fullClassCorpus[c]
        random.shuffle(classList)
        testCount = int(len(classList) * testFraction)
        testClassCorpus.append(classList[:testCount])
        trainClassCorpus.append(classList[testCount:])
    return testClassCorpus, trainClassCorpus

testClassCorpus, trainClassCorpus = splitClassCorpus(fullClassCorpus)

import math

def trainBernoulliNB(trainClassCorpus):
    N = sum(len(classList) for classList in trainClassCorpus) # total number of documents of any class
    classesCount = len(trainClassCorpus)
    pb = progressBar(50)
    pb.start(N)
    V = {}
    i=0
    for c in range(classesCount):
        for text in trainClassCorpus[c]:
            pb.tick(i)
            i += 1
            terms = set([ token.lower() for token in text if token.isalpha() ] )
            for term in terms:
                if term not in V:
                    V[term] = [0] * classesCount
                V[term][c] += 1
    pb.stop()

    Nc = [ len(classList) for classList in trainClassCorpus ]
    prior = [ Nc[c] / N for c in range(classesCount) ]
    condProb = {}
    for t in V:
        condProb[t] = [ (V[t][c] +1) / (Nc[c] + 2) for c in range(classesCount)]
    return condProb, prior, V

condProbB, priorB, VB = trainBernoulliNB(trainClassCorpus)

#? what is features? | i guess select more relevant words
def applyBernoulliNB_SLOW(prior, condProb, text, features = None ):
    terms = set([ token.lower() for token in text if token.isalpha() ] )
    for c in range(len(prior)):
        score = math.log(prior[c])
        for t in condProb:
            if features and t not in features: continue
            if t in terms:
                score += math.log(condProb[t][c])
            else:
                score += math.log(1.0 - condProb[t][c])
        if c == 0 or score > maxScore:
            maxScore = score
            answer = c
    return answer
#? whys that: here we start with an empty doc (our goal is to iterate over the terms in the doc instead over the terms in the whole vocab
def calcInitialCondProb(condProb, features = None):
    classesCount = len(condProb[next(iter(condProb))])
    initialCondProb = [0.0] * classesCount
    for t in features if features else condProb:
        for c in range(classesCount):
            initialCondProb[c] += math.log(1.0 - condProb[t][c])
    return initialCondProb
#? the fast version is iteratin over the terms in the doc instead over the terms in the vocab
def applyBernoulliNB(prior, condProb, initialCondProb, text, features = None ):
    terms = set([ token.lower() for token in text if token.isalpha() ] )
    for c in range(len(prior)):
        score = math.log(prior[c]) + initialCondProb[c]
        for t in terms:
            if t not in condProb: continue
            if features and t not in features: continue
            score += math.log( condProb[t][c] / (1.0 - condProb[t][c]) )
        if c == 0 or score > maxScore:
            maxScore = score
            answer = c
    return answer

print('Първият текст е класифициран с Бернулиев модел като: '+classes[applyBernoulliNB_SLOW(priorB, condProbB, text1.split())])
print('Вторият текст е класифициран с Бернулиев модел като: '+classes[applyBernoulliNB_SLOW(priorB, condProbB, text2.split())])
print('Третият текст е класифициран с Бернулиев модел като: '+classes[applyBernoulliNB_SLOW(priorB, condProbB, text3.split())])

initialCondProbB = calcInitialCondProb(condProbB)
print('Първият текст е класифициран с Бернулиев модел като: '+classes[applyBernoulliNB(priorB, condProbB, initialCondProbB, text1.split())])
print('Вторият текст е класифициран с Бернулиев модел като: '+classes[applyBernoulliNB(priorB, condProbB, initialCondProbB, text2.split())])
print('Третият текст е класифициран с Бернулиев модел като: '+classes[applyBernoulliNB(priorB, condProbB, initialCondProbB, text3.split())])

def trainMultinomialNB(trainClassCorpus):
    N = sum(len(classList) for classList in trainClassCorpus)
    classesCount = len(trainClassCorpus)
    pb = progressBar(50)
    pb.start(N)
    V = {}
    i=0
    for c in range(classesCount):
        for text in trainClassCorpus[c]:
            pb.tick(i)
            i += 1
            terms = [ token.lower() for token in text if token.isalpha() ]
            for term in terms:
                if term not in V:
                    V[term] = [0] * classesCount
                V[term][c] += 1
    pb.stop()

    Nc = [ (len(classList)) for classList in trainClassCorpus ]
    prior = [ Nc[c] / N for c in range(classesCount) ]
    T = [0] * classesCount
    for t in V:
        for c in range(classesCount):
            T[c] += V[t][c]
    condProb = {}
    for t in V:
        condProb[t] = [ (V[t][c] +1) / (T[c] + len(V)) for c in range(classesCount)]
    return condProb, prior, V

condProbM, priorM, VM = trainMultinomialNB(trainClassCorpus)

def applyMultinomialNB(prior, condProb, text, features = None ):
    terms = [ token.lower() for token in text if token.isalpha() ]
    for c in range(len(prior)):
        score = math.log(prior[c])
        for t in terms:
            if t not in condProb: continue
            if features and t not in features: continue
            score += math.log(condProb[t][c])
        if c == 0 or score > maxScore:
            maxScore = score
            answer = c
    return answer

print('Първият текст е класифициран с Мултиномен модел като: '+classes[applyMultinomialNB(priorM, condProbM, text1.split())])
print('Вторият текст е класифициран с Мултиномен модел като: '+classes[applyMultinomialNB(priorM, condProbM, text2.split())])
print('Третият текст е класифициран с Мултиномен модел като: '+classes[applyMultinomialNB(priorM, condProbM, text3.split())])

def testClassifier(testClassCorpus, gamma):
    L = [ len(c) for c in testClassCorpus ]
    pb = progressBar(50)
    pb.start(sum(L))
    i = 0
    classesCount = len(testClassCorpus)
    confusionMatrix = [ [0] * classesCount for _ in range(classesCount) ]
    for c in range(classesCount):
        for text in testClassCorpus[c]:
            pb.tick(i)
            i+=1
            c_MAP = gamma(text)
            confusionMatrix[c][c_MAP] += 1
    pb.stop()
    precision = []
    recall = []
    Fscore = []
    for c in range(classesCount):
        extracted = sum(confusionMatrix[x][c] for x in range(classesCount))
        if confusionMatrix[c][c] == 0:
            precision.append(0.0)
            recall.append(0.0)
            Fscore.append(0.0)
        else:
            precision.append( confusionMatrix[c][c] / extracted )
            recall.append( confusionMatrix[c][c] / L[c] )
            Fscore.append((2.0 * precision[c] * recall[c]) / (precision[c] + recall[c]))
    P = sum( L[c] * precision[c] / sum(L) for c in range(classesCount) )
    R = sum( L[c] * recall[c] / sum(L) for c in range(classesCount) )
    F1 = (2*P*R) / (P + R)
    return confusionMatrix, precision, recall, Fscore, P, R, F1

gamma = lambda text : applyBernoulliNB(priorB, condProbB, initialCondProbB, text)
confusionMatrix, precision, recall, Fscore, P, R, F1 = testClassifier(testClassCorpus, gamma)
print('Матрица на обърквания: ')
for row in confusionMatrix:
    for val in row:
        print('{:4}'.format(val), end = '')
    print()
print('Прецизност: '+str(precision))
print('Обхват: '+str(recall))
print('F-оценка: '+str(Fscore))
print('Обща презизност: '+str(P)+', обхват: '+str(R)+', F-оценка: '+str(F1))
print()

gamma = lambda text : applyMultinomialNB(priorM, condProbM, text)
confusionMatrix, precision, recall, Fscore, P, R, F1 = testClassifier(testClassCorpus, gamma)
print('Матрица на обърквания: ')
for row in confusionMatrix:
    for val in row:
        print('{:4}'.format(val), end = '')
    print()
print('Прецизност: '+str(precision))
print('Обхват: '+str(recall))
print('F-оценка: '+str(Fscore))
print('Обща презизност: '+str(P)+', обхват: '+str(R)+', F-оценка: '+str(F1))
print()
