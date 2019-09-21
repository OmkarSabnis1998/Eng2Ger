from collections import Counter
import NMT_Model
import nmt_data_utils
import nmt_model_utils
import h5py
import keras

with open('M:/Deep and Machine Learning Projects/Eng2Ger Machine Translation/Dataset/europarl-v7.de-en.en','r',encoding='utf-8') as f:
    en = f.readlines()
with open('M:/Deep and Machine Learning Projects/Eng2Ger Machine Translation/Dataset/europarl-v7.de-en.de','r',encoding='utf-8') as f:
    de = f.readlines()
print('Number of English Sentences:')
print(len(en))
print('Number of German Sentences:')
print(len(de))

print('Some of the sentence pairs in the dataset:')
for i in zip(en[:5],de[:5]):
    print(i,'\n')

de = [i.strip() for i in de]
en = [i.strip() for i in en]

len_en = [len(sent) for sent in en if 20 < len(sent) < 50]
len_dist = Counter(len_en).most_common()
print(len_dist)

_de = []
_en = []
for sent_de, sent_en in zip(de, en):
    if 20 < len(sent_en) < 50:
        _de.append(sent_de)
        _en.append(sent_en)

en_preprocessed, en_most_common = nmt_data_utils.preprocess(_en[:5000])
de_preprocessed, de_most_common = nmt_data_utils.preprocess(_de[:5000], language = 'german')
print(len(en_preprocessed), len(de_preprocessed))

en_preprocessed_clean, de_preprocessed_clean = [], []

for sent_en, sent_de in zip(en_preprocessed, de_preprocessed):
    if sent_en != [] and sent_de != []:
        en_preprocessed_clean.append(sent_en)
        de_preprocessed_clean.append(sent_de)
    else:
        continue
print(len(en_preprocessed_clean), len(de_preprocessed_clean))

for e, d in zip(en_preprocessed_clean, de_preprocessed_clean[:5]):
    print('English:\n', e)
    print('German:\n', d, '\n'*3)

specials = ["<unk>", "<s>", "</s>", '<pad>']

en_word2ind, en_ind2word, en_vocab_size = nmt_data_utils.create_vocab(en_most_common, specials)
de_word2ind, de_ind2word, de_vocab_size = nmt_data_utils.create_vocab(de_most_common, specials)

print(en_vocab_size, de_vocab_size)

en_inds, en_unknowns = nmt_data_utils.convert_to_inds(en_preprocessed_clean, en_word2ind, reverse = True, eos = True)
de_inds, de_unknowns = nmt_data_utils.convert_to_inds(de_preprocessed_clean, de_word2ind, sos = True, eos = True)

print([nmt_data_utils.convert_to_words(sentence, en_ind2word) for sentence in  en_inds[:2]])

print([nmt_data_utils.convert_to_words(sentence, de_ind2word) for sentence in  de_inds[:2]])

num_layers_encoder = 4
num_layers_decoder = 4
rnn_size_encoder = 128
rnn_size_decoder = 128
embedding_dim = 300

batch_size = 64
epochs =200
clip = 5
keep_probability = 0.8
learning_rate = 0.01
learning_rate_decay_steps = 1000
learning_rate_decay = 0.9


nmt_model_utils.reset_graph()

nmt = NMT_Model.NMT(en_word2ind,
                    en_ind2word,
                    de_word2ind,
                    de_ind2word,
                    './models/local_one/my_model',
                    'TRAIN',
                    embedding_dim = embedding_dim,
                    num_layers_encoder = num_layers_encoder,
                    num_layers_decoder = num_layers_decoder,
                    batch_size = batch_size,
                    clip = clip,
                    keep_probability = keep_probability,
                    learning_rate = learning_rate,
                    epochs = epochs,
                    rnn_size_encoder = rnn_size_encoder,
                    rnn_size_decoder = rnn_size_decoder, 
                    learning_rate_decay_steps = learning_rate_decay_steps,
                    learning_rate_decay = learning_rate_decay)
  
nmt.build_graph()
nmt.train(en_inds, de_inds)


_de_inds_de_inds, _de_unknowns = nmt_data_utils.convert_to_inds(de_preprocessed_clean, de_word2ind, sos = True,  eos = True)

nmt_model_utils.reset_graph()

nmt = NMT_Model.NMT(en_word2ind,
                    en_ind2word,
                    de_word2ind,
                    de_ind2word,
                    './models/local_one/my_model',
                    'INFER',
                    num_layers_encoder = num_layers_encoder,
                    num_layers_decoder = num_layers_decoder,
                    batch_size = len(en_inds[:50]),
                    keep_probability = 1.0,
                    learning_rate = 0.0,
                    beam_width = 0,
                    rnn_size_encoder = rnn_size_encoder,
                    rnn_size_decoder = rnn_size_decoder)

nmt.build_graph()
preds = nmt.infer(en_inds[:50], restore_path =  './models/local_one/my_model', targets = de_inds[:50])

print(nmt_model_utils.sample_results(preds, en_ind2word, de_ind2word, en_word2ind, de_word2ind, de_inds[:50], en_inds[:50]))

