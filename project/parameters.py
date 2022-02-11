import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc_hid_size = 128
dec_hid_size = 128
enc_dropout = 0.1
dec_dropout = 0.1
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_posf_size = 256
dec_posf_size = 256

limit = 1000
beam_width = 3
alpha = 0.7

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
