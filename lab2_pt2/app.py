from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from gensim.models.doc2vec import TaggedDocument
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import tensorflow as tf
import pandas as pd
import numpy as np

app = Flask(__name__)

df = pd.read_csv("SPAM text message 20170820 - Data (1).csv")
model = tf.keras.models.load_model("Mymodel.h5")

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
df['Message'] = df['Message'].apply(cleanText)
max_fatures = 500000
MAX_SEQUENCE_LENGTH = 50
tokenizer = Tokenizer(num_words=max_fatures, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Message'].values)
X = tokenizer.texts_to_sequences(df['Message'].values)
X = pad_sequences(X)
X = tokenizer.texts_to_sequences(df['Message'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        inp = request.form['input']
        seq = tokenizer.texts_to_sequences([inp])
        padded = pad_sequences(seq, maxlen=X.shape[1], dtype='int32', value=0)
        pred = model.predict(padded)
        print(pred)
        labels = ['ham','spam']
        # print(pred)
        return render_template("index.html", predicted="Detected as: "+labels[np.argmax(pred)], inp="Input Text: "+inp)

if __name__ == "__main__":
    app.run(debug=True)