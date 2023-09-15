import streamlit as st
import requests
from streamlit_lottie import st_lottie
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import nltk
import pandas as pd
import numpy as np
import re
from functools import lru_cache
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

import pickle
import spacy
import contractions as contract

import pkg_resources
from symspellpy import SymSpell
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer



st.set_page_config(page_title="PsychoAI", page_icon=":brain:", layout="wide")


####################
# LOAD MODELS
####################
@st.cache_resource
def lgbm_load():
    lgbm_model = joblib.load(r'C:\Users\K\Desktop\final_directory\lgbm.pkl')
    return lgbm_model

lgbm_model = lgbm_load()

@st.cache_resource
def tf_load():
    tf_model = load_model(r'C:\Users\K\Desktop\final_directory\model_clean.h5',
                              custom_objects={'KerasLayer': hub.KerasLayer})
    return tf_model


tf_model = tf_load()

tfidf = TfidfVectorizer(vocabulary=pickle.load(open(r"C:\Users\K\Desktop\final_directory\feature.pkl", "rb")))



###############################
# DATA PREPROCESS
###############################

@st.cache_data
def load_nlp_and_sym_spell():
    nlp = spacy.load("en_core_web_sm")
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    return nlp, sym_spell

nlp, sym_spell = load_nlp_and_sym_spell()

# fix spelling mistakes and gives suggestions
def fix_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    correctedtext = suggestions[0].term
    return correctedtext


def remove_whitespace(text):
    text = text.strip()
    return ' '.join(text.split())


def remove_url(text):
    return re.sub(r'http\S+', '', text)


def remove_mail(text):
    return re.sub(r'\S+@\S+', '', text)


def remove_symbols_digits(text):
    return re.sub('[^a-zA-Z\s]', ' ', text)


def remove_emoji(text):
    return re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001FB00-\U0001FBFF\U0001FE00-\U0001FE0F\U0001F004]+',
        '', text)


def remove_special(text):
    return text.replace('\r', ' ').replace('\n', ' ').replace('    ', ' ').replace('"', '')


def fix_lengthening(text):
    pattern = re.compile(r'(.)\1{2,}')
    return pattern.sub(r'\1\1', text)


def text_preprocessing(text, contractions=True, convert_num=True,
                       extra_whitespace=True, lemmatization=True, lowercase=True,
                       url=True, symbols_digits=True, special_chars=True,
                       stop_words=True, lengthening=True, spelling=True, emoji=True):


    if lowercase:
        text = text.lower()
    if emoji:
        text = remove_emoji(text)
    if contractions:
        text = contract.fix(text)
    if url:
        text = remove_url(text)
    if symbols_digits:
        text = remove_symbols_digits(text)
    if special_chars:
        text = remove_special(text)
    if extra_whitespace:
        text = remove_whitespace(text)
    if lengthening:
        text = fix_lengthening(text)
    if spelling:
        text = fix_spelling(text)


    doc = nlp(text)  # tokenise text

    clean_text = []

    # return text

    for token in doc:

        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM' and token.text not in ["no", "not"]:
            flag = False
        # exclude number words
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            flag = False
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag == True:
            clean_text.append(edit)
    return " ".join(clean_text)




###################################
# STREAMLIT
###################################

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


url = "https://lottie.host/fa022661-dffb-44fc-a9c8-a455dbef9cd2/355iNkm8mn.json"

lottie_animation = load_lottieurl(url)


with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Psycho AI :brain:")
        st.title("A Suicidal Thought Predictor")

    with right_column:
        st_lottie(lottie_animation, height=300, key="brainloading")


with st.container():
    left_column, right_column = st.columns(2)
    st.subheader("How Do You Feel?")
    text_input = st.text_input("Try to write at least 3 sentences", "")

    if st.button("PREDICT"):
        if text_input:
            text_preprocessed = text_preprocessing(text_input)
            prediction = tf_model.predict([text_preprocessed])[0][0]

            text_tfidf = tfidf.fit_transform([text_preprocessed])
            sentiment_prediction = lgbm_model.predict(text_tfidf).reshape(1, -1)[0][0]

        with st.container():
            st.write(f"%{round(prediction * 100, 2)} Suicidal")
            st.write(f"Sentiment: {sentiment_prediction}")


