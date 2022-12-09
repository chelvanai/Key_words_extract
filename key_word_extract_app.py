import pandas as pd
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import streamlit as st


def get_candidate_words(text, combine1, combine2):
    n_gram_range = (combine1, combine2)
    stop_words = "english"
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
    candidates = count.get_feature_names_out()
    return candidates


def get_keywords(text, candidate_words, model, tokenizer, top_k):
    candidate_tokens = tokenizer(candidate_words.tolist(), padding=True, return_tensors="pt")
    candidate_embeddings = model(**candidate_tokens)["pooler_output"]

    text_tokens = tokenizer([text], padding=True, return_tensors="pt")
    text_embedding = model(**text_tokens)["pooler_output"]

    candidate_embeddings = candidate_embeddings.detach().numpy()
    text_embedding = text_embedding.detach().numpy()

    distances = cosine_similarity(text_embedding, candidate_embeddings)
    keywords = [candidate_words[index] for index in distances.argsort()[0][-top_k:]]
    accuracy = [distances[0][index] for index in distances.argsort()[0][-top_k:]]
    df = pd.DataFrame(list(zip(keywords, accuracy)), columns=['keywords', 'score'])
    df = df.round(2)
    df = df.sort_values(by='score', ascending=False)
    df.index = np.arange(0, len(df))
    return df


model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

with st.form("my_form"):
    st.write("Key words extract")
    text = st.text_area("Paste or write text", height=300)
    combine1 = st.number_input('Word combines 1', value=1)
    combine2 = st.number_input('Word combines 2', value=1)

    top_keywords = st.number_input('Top keywords', value=5)

    submitted = st.form_submit_button("Submit")
    if submitted:
        candidate_words = get_candidate_words(text, combine1, combine2)
        result_df = get_keywords(text, candidate_words, model, tokenizer, top_keywords)
        st.info("Extracted keywords")
        st.table(result_df.style.format({"score": "{:.2f}"}))
