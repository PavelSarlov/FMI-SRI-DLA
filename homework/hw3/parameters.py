import torch

corpusFileName = 'corpusFunctions'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'

device = torch.device("cuda:0")
# device = torch.device("cpu")

batchSize = 64
char_emb_size = 32

hid_size = 128
lstm_layers = 2
dropout = 0.6

epochs = 11
learning_rate = 0.0001

defaultTemperature = 0.4
