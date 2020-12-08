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

st.cache(suppress_st_warning=True)
df = pd_load(baserepo + "tweets.csv")
sentence_sets = []
with open('data/ners.pkl', 'rb') as f:
    sentence_sets = pickle.load(f)

st.cache(suppress_st_warning=True)
tweets = df.text.tolist()
num_tweets = len(tweets)

random.seed(0)
st.subheader("Here are some randomly selected recent tweets about Covid-19")

pics = {"[Tweet #1]": "1261978560249683969.png",
"[Tweet #2]":"1249944985316794372.png",
"[Tweet #3]":"1295438615250472960.png",
"[Tweet #4]":"1256772521774518273.png",
"[Tweet #5]":"1260726271321018372.png"}

pic = st.selectbox("Tweets choices", list(pics.keys()), 0)
st.image(pics[pic], use_column_width=True, width = 600)

# indices = random.sample(range(num_tweets), 5)
indices = [316, 148, 646, 225, 305]
#for i, tweet in enumerate([tweets[i] for i in indices]):
    #st.write("[{}] ".format(i+1) + tweet)


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
#st.write("Here is the tweet you selected!")
#st.write([tweets[i] for i in indices][tweet_option])
st.subheader("How many similar tweets would you like to retrieve?")
n_opt = st.slider("", min_value=1, max_value=5, value=3, step=1)


st.subheader("Now please select your hyperparameters")
learning_rate = float(st.radio("Choose model learning rate", ('1e-5', '2e-5')))
batch_size =int(st.radio("Choose model batch size", ('64', '32')))
epochs =int(st.radio("Choose model number of training epochs", ('10', '5')))

all_scores = load_marix(baserepo + "adjs/adj_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + ".npy")



sorted_row_idx, best_scores = get_top_n_idx(all_scores[indices], n_opt)
sorted_row_idx = sorted_row_idx[tweet_option].tolist()[::-1][1:]
best_scores = best_scores[tweet_option].tolist()[::-1][1:]

st.subheader("Our Model - Sentence Encoder Model Result:")
st.write('Here are the ordered top ' + str(n_opt) + ' tweets similar to this tweet:')

for tweet_idx, score in zip(sorted_row_idx, best_scores):
    st.write(tweets[tweet_idx])
    st.write("with similarity score " + str(1 + np.log(score)))
    st.write("\n")

question = [tweets[i] for i in indices][tweet_option]
doc, tokens, matches = get_best_match_ner(question, n_opt+1)


st.sidebar.write("Model Explanations")

if st.sidebar.button("NER"):
    st.sidebar.write("Named Entity Recognition uses a small Convolutional Neural Network")
    st.sidebar.write("It is trained on generic English web text and performs poorly on certain tweets")
    st.sidebar.write("Each document is broken down into Tokens with several [Attributes](https://spacy.io/api/token#attributes)")
    st.sidebar.markdown("Tweets are matched based on [Jaccard Overlap](https://news.developer.nvidia.com/similarity-in-graphs-jaccard-versus-the-overlap-coefficient/)")
if st.sidebar.button("Sentence Encoder"):
    st.sidebar.markdown("The Sentence Encoder uses Long-Term Short Term - a type of [Recurrent Neural Net designed for sequences](https://arxiv.org/pdf/1705.02364.pdf)")
    st.sidebar.write("It is trained on the Stanford Natural Language Inference Set")
    st.sidebar.write("This transfer learning method generalizes well to new sequences")
    
st.subheader("What are the token attributes of the example tweet?")
Attributes = st.multiselect('Select token attributes to display',[
        "idx",
        "text",
        "lemma_",
        "pos_",
        "tag_",
        "head",
        "ent_type_",
    ], ["text","ent_type_"])
st.write("Click the button to see the result")
if st.button("Show token attributes"):
    attrs = Attributes
    data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)

st.subheader("Baseline Model - NER Model Result:")
st.write('Here are the ordered top ' + str(n_opt) + ' tweets similar to this tweet BY NER tag overlap:')
for tweet_idx in matches:
    if(tweet_idx != indices[tweet_option]):
        st.write(tweets[tweet_idx])
        st.write("\n")

st.write("You can see our fine tuned model performs better than the NER baseline model because the output tweets are more similar")
