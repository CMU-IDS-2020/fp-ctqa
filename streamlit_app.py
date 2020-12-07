# Following architecture file copied from InferSent source code for load pretraining model
# In the later stage, this architecture will be modified to train a binary classifier (not 3 way softmax)
# We use the original code first to make sure the inference pipeline is good
# so that we don't have to worry about output capatibility when re-train this model with Quora

import numpy as np
import time
import pandas as pd
import random
# import torch
# import torch.nn as nn
import streamlit as st
import urllib
import re
import pickle
import requests
import io
# import nltk
# nltk.download('punkt')
# import warnings
# warnings.filterwarnings("ignore")
import spacy
# nlp = spacy.load('en_core_web_sm')
import en_core_web_sm
nlp = en_core_web_sm.load()

baserepo = "https://raw.githubusercontent.com/CMU-IDS-2020/fp-ctqa/main/data/"
st.title('Covid-19 Twitter Search')
def pd_load(inp):
    return pd.read_csv(inp)

def load_marix(inp):
    response = requests.get(inp)
    response.raise_for_status()
    all_scores = np.load(io.BytesIO(response.content))
    return all_scores

df = pd_load(baserepo + "tweets.csv")

tweets = df.text.tolist()
num_tweets = len(tweets)

random.seed(0)
st.subheader("Here are some randomly selected most recent tweets about Covid-19")
indices = random.sample(range(num_tweets), 5)
for i, tweet in enumerate([tweets[i] for i in indices]):
    st.write("[{}] ".format(i+1) + tweet)

sentence_sets = []
with open('data/ners.pkl', 'rb') as f:
    sentence_sets = pickle.load(f)

def get_top_n_idx(A, N):
    N += 1 # not self
    col_idx = np.arange(A.shape[0])[:,None]
    sorted_row_idx = np.argsort(A, axis=1)[:,A.shape[1]-N::]
    best_scores = A[col_idx,sorted_row_idx]
    return sorted_row_idx, best_scores


def get_best_match_ner(question, n_opt):
    def jaccard_overlap(question, sentence):
        if len(question) is 0 or len(sentence) is 0:
            return 1
        intersection = len(question.intersection(sentence))
        union = len(question.union(sentence))
        return float(intersection)/float(union)
    question_doc = nlp(question)
    question_token_set = set([(X.text, X.ent_type_) for X in question_doc])
    index = [jaccard_overlap(question_token_set, sentence_set) for sentence_set in sentence_sets]
    return question_doc, question_token_set, np.argpartition(index, -1 * n_opt)[-1 * n_opt:]

sample_ids = [1,2,3,4,5]

st.subheader("Which tweet would you like to get information on?")
st.write("Please select the tweet id based on the number inside [] above")
tweet_option = st.selectbox('', sample_ids)
tweet_option -= 1
st.write("Here is the tweet you selected!")
st.write([tweets[i] for i in indices][tweet_option])
st.subheader("How many similar tweets would you like to retrieve?")
n_opt = st.slider("", min_value=1, max_value=5, value=3, step=1)


st.subheader("Now please select your hyperparameters")
learning_rate = float(st.radio("Choose model learning rate", ('1e-5', '2e-5')))
batch_size =int(st.radio("Choose model batch size", ('64', '32')))
epochs =int(st.radio("Choose model number of training epochs", ('10', '5')))

all_scores = load_marix(baserepo + "adjs/adj_" + str(batch_size) + "_" + str(learning_rate) + "-" + str(epochs) + ".npy")



sorted_row_idx, best_scores = get_top_n_idx(all_scores[indices], n_opt)
sorted_row_idx = sorted_row_idx[tweet_option].tolist()[::-1][1:]
best_scores = best_scores[tweet_option].tolist()[::-1][1:]


st.write('Here are the ordered top ' + str(n_opt) + ' tweets similar to this tweet:')

for tweet_idx, score in zip(sorted_row_idx, best_scores):
    st.write(tweets[tweet_idx])
    st.write("with similarity score " + str(score))
    st.write("\n")

question = [tweets[i] for i in indices][tweet_option]
doc, tokens, matches = get_best_match_ner(question, n_opt+1)
st.write('Token Attributes')
Attributes = st.multiselect('Select token attributes to display',[
        "idx",
        "text",
        "lemma_",
        "pos_",
        "tag_",
        "dep_",
        "head",
        "ent_type_",
    ], ["text","ent_type_"])
if st.button("Show token attributes"):
    attrs = Attributes
    data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)

st.write('Here are the ordered top ' + str(n_opt) + ' tweets similar to this tweet BY NER tag overlap:')
for tweet_idx in matches:
    if(tweet_idx != indices[tweet_option]):
        st.write(tweets[tweet_idx])
        st.write("\n")
