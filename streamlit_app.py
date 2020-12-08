import altair as alt
import numpy as np
import time
import pandas as pd
import streamlit as st
import urllib
import re
import pickle
import requests
import io
import spacy
# nlp = spacy.load('en_core_web_sm')
import en_core_web_sm
nlp = en_core_web_sm.load()

# wordcloud imports
import wordcloud
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def pd_load(inp):
    return pd.read_csv(inp)

def load_marix(inp):
    response = requests.get(inp)
    response.raise_for_status()
    all_scores = np.load(io.BytesIO(response.content))
    return all_scores

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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Introduction and data description
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
st.title('Covid-19 Twitter Search')
st.header("Introduction")
st.success("Since the outbreak of COVID-19 in early 2020, people have been searching frequently on social media platforms such as Twitter about health issues. \
           The vast amount of tweets has also made it difficult for people to quickly locate their content of interest. \
           Using Twitter hashtags alone to label tweets still leaves an enormous corpus. \
           In this situation, it is natural to ask the following question: how do we help people filter related tweets with more efficiency and flexibility?\n\
           \nWe want to approach this search problem from a different perspective: instead of performing search using keywords\
           , we find it more natural for the users to have the following user story flow: \
           look at a few recent tweets, find the one they are more intersted in, and then dig into similar contents.")

st.header("Simple Explotory Data Analysis")
st.subheader("Data Preprocessing and Distribution")
st.success("[TODO - Yuanyuan] we can include the source of data, sample tweets, and a flow chart for data preprocessing")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Word cloud
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
st.cache(suppress_st_warning=True)
cloud_tweets = pd_load(
    "https://raw.githubusercontent.com/CMU-IDS-2020/a3-a3-formed-group/master/tweets/frequent_terms.csv").reset_index()
min_val = int(cloud_tweets['counts'].min())
st.cache(suppress_st_warning=True)
cloud_tweets = cloud_tweets[cloud_tweets.counts <= 5*min_val]
st.cache(suppress_st_warning=True)
raw_dic = pd.Series(cloud_tweets.counts.values, index=cloud_tweets.term).to_dict()

# load interactivity elements
st.cache(suppress_st_warning=True)
st.subheader('Word Usage in #Covid-19 Tweets')
color_func_twit = wordcloud.get_single_color_func("#00acee")
st.sidebar.write("Choose Word Cloud Options")
remove_eng = st.sidebar.checkbox("Remove English Stop Words")
remove_esp = st.sidebar.checkbox("Remove Spanish Stop Words")
show_chart = st.button('Show Distribution')
slider_ph = st.empty()
value = slider_ph.slider("Choose Max Frequency", min_value=min_val,
                         max_value=5*min_val, value=2*min_val, step=10)

# user text input
custom = st.sidebar.text_input('Add Custom Stopwords (comma separated)')
custom = custom.split(',')

lemma = st.sidebar.checkbox("Lemmatize")

# create stopwords list
st.cache(suppress_st_warning=True)
stop_words = []
if(custom):
    stop_words += custom
if(remove_eng):
    stop_words += stopwords.words('english')
if(remove_esp):
    stop_words += stopwords.words('spanish')


# create chart
st.cache(suppress_st_warning=True)
basic_chart = alt.Chart(cloud_tweets[cloud_tweets['counts'] <= value]).mark_bar().encode(
    x=alt.X('index', title='Rank in Corpus'),
    y='counts'
).interactive()


# create lemmatized dictionary
st.cache(suppress_st_warning=True)
lemmatizer = WordNetLemmatizer()
lemma_dic = {lemmatizer.lemmatize(k.strip()): v for k, v in raw_dic.items()}

# choose dictionary to generate wordcloud
st.cache(suppress_st_warning=True)
if(lemma):
    dic = {k: v for k, v in lemma_dic.items(
    ) if v <= value and k not in stop_words}
    st.sidebar.write("Words will be Lemmatized")
    st.sidebar.markdown("[More Info (External Link)](https://en.wikipedia.org/wiki/Lemmatisation)")

else:
    dic = {k: v for k, v in raw_dic.items() if v <= value and k not in stop_words}

# create wordcloud
st.cache(suppress_st_warning=True)
if(any(dic)):
    wc = wordcloud.WordCloud(color_func=color_func_twit).generate_from_frequencies(frequencies=dic)
    fig = plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(fig)
    if(show_chart):
        st.altair_chart(basic_chart)
else:
    st.write("All words have been filtered out. Try removing Stopwords.")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# User select tweets
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
baserepo = "https://raw.githubusercontent.com/CMU-IDS-2020/fp-ctqa/main/data/"
df = pd_load(baserepo + "tweets.csv")
tweets = df.text.tolist()
num_tweets = len(tweets)
indices = [316, 148, 646, 225, 305]

st.subheader("Please select one recent tweet about Covid-19 and start your exploration")
pics = {"[Tweet #1]": "pictures/1261978560249683969.png",
"[Tweet #2]":"pictures/1249944985316794372.png",
"[Tweet #3]":"pictures/1295438615250472960.png",
"[Tweet #4]":"pictures/1256772521774518273.png",
"[Tweet #5]":"pictures/1260726271321018372.png"}
pic = st.selectbox("", list(pics.keys()), 0)
st.image(pics[pic], use_column_width=True, width = 600)

sample_ids = [1,2,3,4,5]
tweet_option = int(pic[-2]) - 1
st.subheader("How many similar tweets would you like to retrieve?")
n_opt = st.slider("", min_value=2, max_value=6, value=3, step=1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# NER Model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# st.sidebar.write("Model Explanations")
# if st.sidebar.button("NER"):
#     st.sidebar.write("Named Entity Recognition uses a small Convolutional Neural Network.")
#     st.sidebar.write("It is trained on generic English web text and performs poorly on certain tweets")
#     st.sidebar.write("Each document is broken down into Tokens with several [Attributes](https://spacy.io/api/token#attributes)")
#     st.sidebar.markdown("Tweets are matched based on [Jaccard Overlap](https://news.developer.nvidia.com/similarity-in-graphs-jaccard-versus-the-overlap-coefficient/)")
# if st.sidebar.button("Sentence Encoder"):
#     st.sidebar.markdown("The Sentence Encoder uses Long-Term Short Term - a type of [Recurrent Neural Net designed for sequences](https://arxiv.org/pdf/1705.02364.pdf)")
#     st.sidebar.write("It is trained on the Stanford Natural Language Inference Set")
#     st.sidebar.write("This transfer learning method generalizes well to new sequences")

sentence_sets = []
with open('data/ners.pkl', 'rb') as f:
    sentence_sets = pickle.load(f)

question = [tweets[i] for i in indices][tweet_option]
doc, tokens, matches = get_best_match_ner(question, n_opt+1)


st.header("Baseline Model - NER Model")
st.success("Our baseline Named Entity Recognition model uses a small Convolutional Neural Network. \
           It is trained on generic English web text and performs poorly on certain tweets. \
           Each document is broken down into Tokens with several [Attributes](https://spacy.io/api/token#attributes). \
           Tweets are matched based on [Jaccard Overlap](https://news.developer.nvidia.com/similarity-in-graphs-jaccard-versus-the-overlap-coefficient/). \
           We also provided the users with some explorations on which token attributes contribute most to the similarity score.")
st.write('Here are the ordered top ' + str(n_opt) + ' tweets similar to this tweet BY NER tag overlap:')
for tweet_idx in matches:
    if(tweet_idx != indices[tweet_option]):
        st.info(tweets[tweet_idx])
        st.write("\n")

st.subheader("Which token attributes contribute most to the similarity score?")
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
st.write("Click the button to see the result")
if st.button("Show token attributes"):
    attrs = Attributes
    data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Sentence Encoder
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

st.header("Our Model - Setence Encoder Model")
st.success("We use Facebook Infersent Natural Language Inference model to encode the the tweets to 4096-way vectors\
            and then use vector level cosine similarity scoring to compute top-n similar tweets related to one tweet.\
            An overview of Natural Language Inference architre is as follows and you can find more [more details here](https://research.fb.com/wp-content/uploads/2017/09/emnlp2017.pdf). \
            We train the whole classification model end to end but only use the weights up to sentence encoder part to encode the tweets. \
            \n\nFor training set, we use [Quora Question Pair dataset](https://www.kaggle.com/c/quora-question-pairs) and [SNLI dataset](https://nlp.stanford.edu/projects/snli/) with highly relevant to daily life / common sense sentence pairs, \
            which is similar to twitter content. We also particularly pick the [Twitter Corpus version of the Glove](http://nlp.stanford.edu/data/glove.twitter.27B.zip) word embedding to improve the accuracy of our model.")

st.image("pictures/nli.png", width = 550, caption='Infersent Architecture')
st.subheader("Now please select your hyperparameters")
learning_rate = float(st.radio("Choose Model Learning Rate", ('1e-5', '2e-5')))
batch_size =int(st.radio("Choose Model Batch Size", ('64', '32')))
epochs =int(st.radio("Choose Model Number of Training Epochs", ('10', '5')))
all_scores = load_marix(baserepo + "adjs/adj_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(epochs) + ".npy")
sorted_row_idx, best_scores = get_top_n_idx(all_scores[indices], n_opt)
sorted_row_idx = sorted_row_idx[tweet_option].tolist()[::-1][1:]
best_scores = best_scores[tweet_option].tolist()[::-1][1:]

st.subheader("Our Model - Sentence Encoder Model Result:")
st.write('Here are the ordered top ' + str(n_opt) + ' tweets similar to this tweet:')

for tweet_idx, score in zip(sorted_row_idx, best_scores):
    display_score = round(float(1 + np.log(score)), 5)
    st.info(tweets[tweet_idx] + "\n\n [similarity score: " + str(display_score) + "]")
    st.write("\n")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Conclusions and future work
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
st.header("Conclusion and Analysis")
st.write("You can see our fine tuned model performs better than the NER baseline model because the output tweets are more similar")
