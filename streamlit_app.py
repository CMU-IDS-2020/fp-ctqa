# Following architecture file copied from InferSent source code for load pretraining model
# In the later stage, this architecture will be modified to train a binary classifier (not 3 way softmax)
# We use the original code first to make sure the inference pipeline is good
# so that we don't have to worry about output capatibility when re-train this model with Quora

import numpy as np
import time
import torch
import torch.nn as nn
import streamlit as st
import urllib
import re
import pickle
import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")


class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2
        return emb

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def load_w2v(self, w2v_path):
        f = urllib.request.urlopen(w2v_path)
        self.word_vec = pickle.load(f)
        f.close()

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        f = urllib.request.urlopen(self.w2v_path)

        for line in f:
            line = line.decode("utf-8")
            word, vec = line.split(' ', 1)
            if k <= K:
                word_vec[word] = np.fromstring(vec, sep=' ')
                k += 1
            else:
                break
        f.close()
        if self.eos not in word_vec:
            word_vec[self.eos] = np.mean(np.stack(word_vec.values(), axis=0), axis=0)
        if self.bos not in word_vec:
            word_vec[self.bos] = np.mean(np.stack(word_vec.values(), axis=0), axis=0)
        with open('glove.pickle', 'wb') as f:
            pickle.dump(word_vec, f, protocol=pickle.HIGHEST_PROTOCOL)
        return word_vec

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
        return torch.FloatTensor(embed)

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx:stidx + bsize])
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]
        return embeddings

def build_nli_net():
  MODEL_PATH = 'https://github.com/CMU-IDS-2020/fp-ctqa/raw/main/infersent2.pkl'
  params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                  'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
  infersent = InferSent(params_model)
  infersent.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_PATH))
  return infersent

def text_prepare(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;#]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    text = str(text)
    text = " ".join([word for word in text.split(" ") if re.search('[a-zA-Z]', word)])
    text = text.lower()
    # text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    # text = re.sub(BAD_SYMBOLS_RE, "", text)
    text = re.sub(' +', ' ', text)
    return text

def cosine(u, v):
  # compute the similarity between two embeddings
  # u and v are matrices!
    result = np.einsum('ij,ij->i', u, v) / ((np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)))
    return np.log(result) + 1

# Inference
infersent = build_nli_net()
infersent.load_w2v("https://github.com/CMU-IDS-2020/fp-ctqa/raw/main/glove.pickle")
# infersent.set_w2v_path("https://github.com/CMU-IDS-2020/fp-ctqa/raw/main/glove.6B.300d.txt")
# infersent.build_vocab_k_words(K=75000)


# Dashboard
st.title("The Most Creative Title For This Project")
st.write("NOT FINAL SUBMISSION | SHARING PIPELINE TEST ONLY")
# Very very simple tweet scoring example
tweet_1 = "Since the start of the pandemic, a total 65 WHO staff stationed in Geneva - working from home and onsite - have tested positive for #COVID19. We have not yet established whether any transmission has occurred on campus, but are looking into the matter."
tweet_2 = "WHO staff who were confirmed positive with #COVID19 in Geneva have received the necessary medical attention. WHO carried out full contact tracing and related protocols. Enhanced cleaning protocols were implemented in relevant offices."
tweet_3 = "Any tweets only my own views. More Guns,Less Crime (Univ Chicago Press, 3rd ed);10 books, 100+academic articles. PhD Econ, Advisor for Research & Science #USDOJ"
st.write("Premise tweet\n")
st.write(tweet_1)
st.write("Hypothesis tweet 1\n")
st.write(tweet_2)
st.write("Hypothesis tweet 2\n")
st.write(tweet_3)
st.write("The similarity score between premise and hypothesis 1 is:")
st.write(cosine(infersent.encode([text_prepare(tweet_1)]), infersent.encode([text_prepare(tweet_2)])).tolist()[0])
st.write("The similarity score between premise and hypothesis 2 is:")
st.write(cosine(infersent.encode([text_prepare(tweet_1)]), infersent.encode([text_prepare(tweet_3)])).tolist()[0])

id_to_text = {}
sample_ids = [1322748291646476288, 1322748291751227397, 1322748291839262723, 1322748292384645120, 1322748296432144385, 1322748297736454144, 1322748298797682690, 1322748301842747392, 1322748304946614273, 1322748305898573825]
for n in sample_ids:
    optimal_tweets = []
    for i in range(5):
        # optimal_tweets.append(get_t_value(n, index)) # This can be a function.
        optimal_tweets.append(str(n) + " has its text as number " + str(i+1))
    id_to_text[n] = optimal_tweets
metric_choices = [id_to_text[name_id] for name_id in sample_ids]

tweet_option = st.selectbox('Which tweet would you like to get information on?', sample_ids)
n_opt = st.selectbox('How many similar tweets would you like to retrieve?', (1, 2, 3, 4, 5))
st.write('Here are the top ' + str(n_opt) + ' tweets similar to this tweet:')
for i in range(n_opt):
    st.write(id_to_text[tweet_option][i])
# After all tasks
# del infersent.word_vec
