import os 
os.chdir("C:\\Users\\vootl\\Documents\\Test Project App")
import pandas as pd
import numpy as np
import re
import wordninja
import nltk
import matplotlib.pyplot as plt 
import seaborn as sns
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import *
import warnings 
# import pandas_profiling
warnings.filterwarnings("ignore", category=DeprecationWarning)

def clean_dollar(s):                                    #### Remove words starting with $ (Company names)
    p = r"\$[\w]*"
    r = re.findall(p, s)
    
    for i in r:
        s = re.sub("\\"+i, '', s)
    return s    
def clean_hash(s):                                      #### Clean text in hash 
    p = r"#[\w]*['?[\w]*]*"                             #### eg:- #ownitdon'ttradeit --> phrases in hashed text carry crucial informationabou the sentiment
    r = re.findall(p, s)                                #### People generally tend to use hash to represent entities, reactions, sentiments, opinions.

    for i in r:                                         #### Hence it is crucial not to remove them and instead extract the joined text and split them 
        s = re.sub(i, ' '.join(wordninja.split(i)), s)  
    return s    


def clean_negation(s):                                  ##### Negatoin indicators carry important information about the intent
    p = r"[A-Za-z]*n't"                                 ##### It's crucial to keep them
    r = re.findall(p, s)
    d = {"shan't":"should not", "won't":"will not", "ain't":"am not", "can't":"can not"}
                                                        ##### Not is sometimes written in short form, we can just replace "n't" to " not" except for these 4 short forms
    for i in r:
        if i in d.keys():
            s = re.sub(i, d[i], s)
        else:
            s = re.sub(i, i[:-3]+" not", s)
    return s

def get_key(num, percent=False):
    num = float(num)
    key = "increase" if num >0 else "decrease" if num<0 else ""

    if abs(num) == 0:
        return ""

    return key if not percent else "percent "+key

def clean_numbers(s):                                   
    s = s.replace(" %", "%")
#     print(s)
    p = r"[\+-][0-9]+\.?[0-9]*%"
    r = re.findall(p, s)
#     print(r)
    for i in r:
#         print(i)
        s = s.replace(i, get_key(i.strip()[:-1], True))

    p = r"[\+-][0-9]+\.?[0-9]*"
    r = re.findall(p, s)
#     print(r)
    for i in r:
#         print(i)
        s = s.replace(i, get_key(i.strip()))
    
    return s

def clean_punc(s):
    p = "[^a-zA-Z]"
    r = re.findall(p, s)
    for i in list(set(r)):
        s = s.replace(i, " ")
    return s
df = pd.read_json("Microblog_Trainingdata.json.txt")
# df.to_csv("data.csv", index=False, header=True)
# print(df.info())

# #### 0. Remove Unnecessary columns
# df.drop(["id", "source", "cashtag"], axis=1, inplace=True)

#### 1. Join Sentences
df["joined"]=df.spans.apply(lambda x: ' '.join([i.strip() for i in x]))

#### 2. Drop rows with empty Strings
# print(df.shape)
df = df.loc[df.joined != '', :].reset_index(drop=True).copy(deep=True)
# print(df.shape)
df.sort_values(by='joined', inplace=True)
# df.to_csv("data.csv", index=False, header=True)

#### 3. Remove rows with Duplicate strings
df = df.groupby("joined")["sentiment score"].mean().reset_index().copy(deep=True)

#### 4. Remove Company Tags;  Convert text to lower case
df["comp_removed"] = df["joined"].apply(clean_dollar)
df["comp_removed"] = df["comp_removed"].str.lower()
#df
#### 5. Remove hash(#) & split the ccombined words in hashtag
df['hash_corrected'] = df["comp_removed"].apply(clean_hash)

#### 6. Negation Preservation before removing Stop Words
df['negate_preserved'] = df["hash_corrected"].apply(clean_negation)
#### 7. Remove Stop Words 
stops = [i for i in stopwords.words('english') if i not in ['not', 'high', 'low', 'up', 'down', 'above', 'below', 'under', 'over']]   #too, very can be used if an LSTM is used
df["stops_removed"] =  pd.Series([ " ".join([j for j in word_tokenize(i) if j not in stops]) for i in df["negate_preserved"]])

#### 8. Replace & encode numeric changes 
df['number_removed'] = df['stops_removed'].str.replace(" %", "%")
df['number_removed'] = df["number_removed"].apply(clean_numbers)

#### 9. Remove punctuation, special characters &  REMOVE EXTRA SPACES
df['punc_removed'] = df['number_removed'].apply(clean_punc)
df['punc_removed'] = df['punc_removed'].apply(lambda x: re.sub(' +', ' ', x))
df = df.groupby("punc_removed")["sentiment score"].mean().reset_index().copy(deep=True)

#### 10. Tokenize, Stem, Lemmatize
df['tokens'] = pd.Series([ [j for j in word_tokenize(i)] for i in df["punc_removed"]])
stemmer = porter.PorterStemmer()
lemmatizer = WordNetLemmatizer()
df['stemmed_tokens'] = df["tokens"].apply(lambda x: [stemmer.stem(i) for i in x]) 
df['lemmatized_tokens'] = df["tokens"].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])

df['stemmed_blog'] = df["stemmed_tokens"].apply(lambda x: ' '.join(x))
df['lemmatized_blog'] = df["lemmatized_tokens"].apply(lambda x: ' '.join(x))

df = df.loc[(df["sentiment score"] != 0) & (df["punc_removed"]!='') & (df["punc_removed"]!=' '), :].reset_index(drop=True).copy(deep=True)

#### 10. Create Polarity flag for Clasification
df["sentiment_polarity"] = 0
df.loc[df["sentiment score"]>0, "sentiment_polarity"] = 1

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import asarray
#from gensim.models.fasttext import FastText
d = df.drop(["sentiment score"], axis=1, inplace=False).copy(deep=True)


max_features = 1000
bow_extractor = CountVectorizer(ngram_range = (1,5), max_df=0.90, min_df=4, max_features=max_features )
bow = bow_extractor.fit_transform(d['lemmatized_blog'])
#bow.shape

#bow_extractor.transform(["he is a boy"])


tfidf_extractor = TfidfVectorizer(ngram_range = (1,5), max_df=0.90, min_df=4, max_features=max_features)
tfidf = tfidf_extractor.fit_transform(d['lemmatized_blog'])
#tfidf.shape

max_features = bow.shape[1]
num_samples = d.shape[0]


def generate_aggregate_features(source, sentences_list, vocab_list, feature_size):
    # source            - Any data structure(model/ dict etc) that allows lookup with the word name and returns a numpy array of embeddings
    # sentences_list    - List of list of tokens (List of sentences, each sentence is a list of tokens)
    # vocab_list        - List of unique words in the training data ( only those words whose embedding vectors are available )
    #                     Even with self trained model, all words won't have embeddings because 
    #                     we restrict training on words with a minimum occurance number to avoid overfiitng
    # feature_size      - Size of each word embedding vector 

    features = np.zeros((len(sentences_list), feature_size))

    for i in range(len(sentences_list)):
        doc_agg_vec = np.zeros((1, feature_size))
        num_matched_words = 0
        for word in sentences_list[i]:
            if word in vocab_list:
                doc_agg_vec += source[word].reshape((1, feature_size))
                num_matched_words += 1
            # else:
                # print(word)
        if num_matched_words > 0 :
            features[i, :] = doc_agg_vec / float(num_matched_words)

    return pd.DataFrame(features)
#
feature_size = 300
#w2v_selftrained_model = gensim.models.Word2Vec(
#    d.lemmatized_tokens, size=feature_size, window=5,
#    min_count=2, negative = 10, workers= 2, seed = 5)
#
#w2v_selftrained_model.train(d.lemmatized_tokens, total_examples= num_samples, epochs=30)
#w2v_selftrained_vocab = list(w2v_selftrained_model.wv.vocab.keys())
#
#w2v_selftrained_features = generate_aggregate_features(source = w2v_selftrained_model,
#                                                       sentences_list = d.lemmatized_tokens.to_list(), 
#                                                       vocab_list = w2v_selftrained_vocab,
#                                                       feature_size = feature_size)

import os
if "glove.6B.300d.txt" not in os.listdir():
    os.system("wget http://nlp.stanford.edu/data/glove.6B.zip")
    os.system("unzip -q glove.6B.zip")

t = Tokenizer()
t.fit_on_texts(d.lemmatized_blog.values.tolist())
vocab_size = len(t.word_index) + 1

w2v_pretrained_lookup = {}

f = open('glove.6B.300d.txt', encoding="utf8")
for i in f:                              # Extract the pre trained embeddings of  words present in our Vocabulary
    x = i.split()
    w = x[0]

    if w.lower() in t.word_index:
        w2v_pretrained_lookup[w] = asarray(x[1:], dtype='float32')

f.close()

w2v_pretrained_features = generate_aggregate_features(source = w2v_pretrained_lookup,
                                                       sentences_list = d.lemmatized_tokens.to_list(), 
                                                       vocab_list = list(w2v_pretrained_lookup.keys()),
                                                       feature_size = feature_size)



####################  Word Level
feature_size_pre = 300

t = Tokenizer()
t.fit_on_texts(d.lemmatized_blog.values.tolist())
vocab_size = len(t.word_index) + 1

encoded_docs = t.texts_to_sequences(d.lemmatized_tokens.values.tolist())
# print(encoded_docs)
max_length = 16

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)

wordvec_arrays_pre_ = np.zeros((vocab_size, feature_size_pre))

for i in t.word_index.keys():
    try:
        wordvec_arrays_pre_[t.word_index[i],:] = np.array(w2v_pretrained_lookup[i]).reshape((1, feature_size_pre))
    except KeyError:
        wordvec_arrays_pre_[t.word_index[i],:] = np.zeros((1, feature_size_pre))


############################################################################################################



######################## This will be used for cleaning new unseen texts
def clean_text(t):
    if t == '':
        return -1
    s = t.lower()
    s = clean_dollar(s)
    s = clean_hash(s)
    s = clean_negation(s)
    
    s = " ".join([j for j in word_tokenize(s) if j not in stops]) 
    s = clean_numbers(s.replace(" %", "%"))
    s = clean_punc(s)
    s = re.sub(' +', ' ', s)
    s = word_tokenize(s)
    s = [lemmatizer.lemmatize(i) for i in s]
    
    if t == '':
        return -1
    
    return s



model_dict = {'Logistic Regression':'LR', "SVM":'SVM', "Random Forest":"RF", "ANN":"NN","CNN":"CNN","RNN":"RNN"}

feature_dict = {'Bag of Words':'BoW', "TfIdf":'TfIdf', "PreTrained Word Embeddings":"w2vpre", 
              "SelfTrained Word Embeddings":"w2vself","Doc to Vector":"d2v"}

def get_features(l, f, word_level=False):
    if f == 'BoW':
        return bow_extractor.transform(l).toarray()
    if f == 'TfIdf':
        return tfidf_extractor.transform(l).toarray()
#    if f == 'w2vself':
#        return generate_aggregate_features(source = w2v_selftrained_model,
#                                    sentences_list = [l],
#                                    vocab_list = w2v_selftrained_vocab,
#                                    feature_size = feature_size)
    if f == 'w2vpre' and not word_level:
        fil = open('glove.6B.300d.txt', encoding="utf8")
        num_words = 0
        temp_vec = np.zeros((1, feature_size))
        for i in fil:                              # Extract the pre trained embeddings of  words present in our Vocabulary
            x = i.split()
            w = x[0]
            if w.lower() in l:
                temp_vec += asarray(x[1:], dtype='float32').reshape((1, feature_size))
                num_words += 1
        fil.close()
        return temp_vec / float(num_words)
    
    if f == 'w2vpre' and word_level:
        temp_vec = np.zeros((max_length, feature_size_pre))
        fil = open('glove.6B.300d.txt', encoding="utf8")
        for i in fil: 
            x = i.split()
            w = x[0]
            if w.lower() in l:
                temp_vec[l.index(w.lower()), :] += asarray(x[1:], dtype='float32').reshape((1, feature_size_pre))
        fil.close()
        
        return temp_vec

    
############################################################################################################
import streamlit as st
from keras.models import load_model
import pickle

def select(txt, opt):
    if text == '':
        return [0,0]
    
    mod, feat = model_dict[opt.split(" - ")[0]], feature_dict[opt.split(" - ")[1]]
    
    
    path = "./Models/"+mod+"_"+feat
    
    if mod in ["NN", "CNN", "RNN"]:
        path += ".h5"
        model = load_model(path)
    else:
        path += ".sav"
        with open(path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
            
    print("----------------------------------------------------------------------------------------")
    print(mod, feat)
    print(path)
    print("----------------------------------------------------------------------------------------")
    
    
    x = clean_text(txt)
    x = get_features(x, feature_dict[opt.split(" - ")[1]], True if mod in ["CNN", "RNN"] else False)
    return model.predict_proba(x)[0].tolist()
    
text = st.text_input(label="Enter Text")

option = st.selectbox('Select a Model', ["Logistic Regression - Bag of Words",
                                         "Logistic Regression - TfIdf",
                                         "Logistic Regression - PreTrained Word Embeddings", 
                                         "SVM - Bag of Words",
                                         "SVM - TfIdf",
                                         "SVM - PreTrained Word Embeddings",
                                         "Random Forest - Bag of Words",
                                         "Random Forest - TfIdf",
                                         "Random Forest - PreTrained Word Embeddings",
                                         "ANN - Bag of Words",
                                         "ANN - TfIdf",
                                         "ANN - PreTrained Word Embeddings"])

    
    
    
plt.figure(figsize=(6,3))
plt.barh(['Negative', 'Positive'], select(text, option), color='rg')
plt.xticks(np.linspace(0, 1, 11))
st.pyplot()
