import os
import importlib
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

app = Flask(__name__)

### load loggistic classification model
loggistic_classifier=joblib.load('loggistic_classifier.pkl')

### upload csv files
standardize_text=pd.read_csv("standardize_Abstract.csv")

### create dictionary
class1_id_df = standardize_text[["class1","class1_id"]].sort_values("class1_id")
id_to_class1 = dict(standardize_text[["class1_id","class1"]].values)

### create stop words
from nltk.corpus import stopwords
stop_nltk = stopwords.words('english')
stop_research = ['proposal', 'propose', 'proposed', 'involve', 'public', 'health',
                  'significance', 'significant', 'goal', 'research', 'relevant',
                  'relevance', 'project', 'narrative', 'study', 'work', 'understand',
                  'result', 'statement', 'address', 'may', 'use', 'unreadable',
                  'investigate', 'investigation', 'also', 'provide', 'new', 'novel',
                  'application', 'approach', 'approximately', 'associated', 'because',
                  'abstract','background','introduction','aim','objective']
stop = stop_nltk + stop_research

### create tfidf vectorizer
all_text = standardize_text['Abstract'].tolist()

### Lenmatization and stemming
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

def get_tokens_stems(text, tokenizer = tokenizer,
                     stem = True, stemmer = stemmer,
                     lemma = False, lemmatizer = lemmatizer):

    #get tokens
    all_text = tokenizer.tokenize(text)

    if lemma:
        #get word lemmas
        all_text = [lemmatizer.lemmatize(word) for word in all_text]

    if stem:
        #get word stems
        all_text = [stemmer.stem(word) for word in all_text]

    return all_text

def map_words_to_stems(list_words, list_roots):

    '''
    Given a list of words and list of lemmas/stems,
    return a dictionary with the lemma/stem as keys
    and a list of words with that lemma/stem as values
    '''

    dict_roots = {}
    for word, root in zip(list_words, list_roots):

        #we don't need to store the word if it is not stemmed
        if root != word:

            #if the lemma/stem is already a key in the dictionary
            if root in dict_roots:

                #if the word is not already listed
                if word not in dict_roots[root]:
                    dict_roots[root].append(word)
            else:
                #add lemma/stem as key and word as value
                dict_roots[root] = [word]

    return dict_roots

for text in all_text:

    #get list of tokenized words and lemmas/stems
    #for each abstract
    list_words = get_tokens_stems(text, stem = False)
    list_lemmas = [lemmatizer.lemmatize(word) for word in list_words]
    list_stems = [stemmer.stem(word) for word in list_words]

dict_lemmas = map_words_to_stems(list_words, list_lemmas)
dict_stems = map_words_to_stems(list_words, list_stems)

tfidf = TfidfVectorizer(min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 1), stop_words=stop,
                       tokenizer = get_tokens_stems)
features = tfidf.fit_transform(all_text).toarray()

### define a few functions
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

def prediction(text):
    df = pd.DataFrame({"text":[text]})
    df = standardize_text(df, "text")

    Abs_tfidf = tfidf.transform(df.text)
    Abs_class = loggistic_classifier.predict(Abs_tfidf)
    Abs_class_name = id_to_class1[int(Abs_class)]
    Abs_class_name_df = pd.DataFrame({"Conference Section": [Abs_class_name]})

    return Abs_class_name_df


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = {}
    if request.method == "POST":
        # get url that the user has entered
        try:
            Abstract = request.form['input']
            results = prediction(Abstract)

            print(results)
            print(type(results))
            results = results.to_html(index = False)
            print(results)

        except:
            errors.append(
                ""
            )
        #r = requests.get(symptoms)
        #print(r.text)

    return render_template('index.html', errors=errors, results=results)


if __name__ == '__main__':
    app.run(debug=True)
