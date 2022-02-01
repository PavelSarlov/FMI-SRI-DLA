import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel_lstm'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc_embed_size = 64
dec_embed_size = 64
enc_hid_size = 128
dec_hid_size = 128
enc_dropout = 0.2
dec_dropout = 0.2

uniform_init = 0.1
learning_rate = 0.0005
clip_grad = 5.0
learning_rate_decay = 0.5

batchSize = 32

maxEpochs = 2
log_every = 10
test_every = 2000

max_patience = 5
max_trials = 5
