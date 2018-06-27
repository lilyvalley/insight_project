import os
import importlib
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.models import model_from_json

from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from keras import backend as K

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend

set_keras_backend("theano")

app = Flask(__name__)

model = model_from_json(open('./data/model_architecture.json').read())
model.load_weights('data/model_trainable.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('data/classes.npy')
word_df = pd.read_csv("data/word_index.csv", index_col=0)
word_index = word_df.to_dict()['0']

def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def test_prediction(text):
    df = pd.DataFrame({"text":[text]})
    df = standardize_text(df, "text")

    tokenizer = RegexpTokenizer(r'\w+')
    tokenizer.tokenize

    df["tokens"] = df["text"].apply(tokenizer.tokenize)

    seq = [word_index[tokens] for tokens in df["tokens"][0]]
    input_data = pad_sequences([seq], maxlen=46)

    test_res = model.predict(input_data)

    top_results_index = test_res.argsort()[0][-5:]
    top_results_ordered = list(reversed(top_results_index))

    top_results = label_encoder.inverse_transform([top_results_ordered])
    top_prob = test_res[0][top_results_ordered]
    outdf = pd.DataFrame({"Symptoms":top_results[0],"Probability":top_prob})
    return outdf[["Symptoms",'Probability']]


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        # get url that the user has entered
        try:
            symptoms = request.form['symptoms']
            results = test_prediction(symptoms)

            print(results)
            if results['Probability'][0] < 0.2:
                errors.append(
                    "The symptoms you entered looks prettly ambigous. You may want to enter something more specific!!"
                )
            results = results.to_html(index = False, classes=["border"])

        except:
            errors.append(
                "The symptoms you entered may have typos or uncommon words. Please enter again"
            )
        #r = requests.get(symptoms)
        #print(r.text)

    return render_template('index.html', errors=errors, results=results)


if __name__ == '__main__':
    app.run(debug=True)
