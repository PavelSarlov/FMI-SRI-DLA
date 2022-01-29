import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


enc_layers = 2
dec_layers = 2
enc_embed_size = 32
dec_embed_size = 32
enc_dropout = 0.5
dec_dropout = 0.5
hidden_size = 64

uniform_init = 0.1
learning_rate = 0.001
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 16

maxEpochs = 2
log_every = 10
test_every = 2000

max_patience = 5
max_trials = 5
