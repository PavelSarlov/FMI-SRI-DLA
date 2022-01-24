import torch

corpusFileName = 'corpusFunctions'
modelFileName = 'modelLSTM'
trainDataFileName = 'trainData'
testDataFileName = 'testData'
char2idFileName = 'char2id'

device = torch.device("cuda:0")
# device = torch.device("cpu")

batchSize = 32 
char_emb_size = 64

hid_size = 128 
lstm_layers = 2
dropout = 0.1

epochs = 8
learning_rate = 0.001

defaultTemperature = 0.4
