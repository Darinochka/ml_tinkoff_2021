import random
from collections import Counter
from collections.abc import Sequence
import pickle
import pandas as pd
import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer


class Document:
    def __init__(self, doc):
        # можете здесь какие-нибудь свои поля подобавлять
        self.url = doc['url']
        self.title = doc['title']
        self.text = doc['body']
        self.vector_title = doc['vectors_title']
        self.vector_body = doc['vectors_body']

        self.count_words = Counter(doc['words_title'] + doc['words_title'])

    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...', self.url]
    
    def __equal__(self, other):
        return self.url == other.url
    
    def __ne__(self, other):
        return not self == other

    def count_word(self, word):
        return self.count_words[word]


class Index(Sequence):
    def __init__(self, vocab):
        self.documents = list()
        self.index = {word: set() for word in vocab}

    def append(self, doc: Document):
        self.documents.append(doc)
        self.update_index(doc)

    def update_index(self, doc):
        for word in doc.count_words:
            self.index[word].add(len(self.documents) - 1)

    def __getitem__(self, w: str) -> set:
        return sorted(self.index[w], key=lambda x: self.documents[x].count_word(w))
    
    def __len__(self):
        return len(self.index)
    
    def get_doc(self, idx):
        return self.documents[idx]

documents = pd.read_csv('data.csv')

def get_tfidf(type_tfidf):
    dummy_fun = lambda doc: doc
    tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)

    if type_tfidf == 'body':
        tfidf.fit(documents.words_body)
    elif type_tfidf == 'title':
        tfidf.fit(documents.words_title)

    return tfidf

tfidf_body = get_tfidf('body')
tfidf_title = get_tfidf('title') 
index = Index(
    list(tfidf_body.vocabulary_.keys()) + list(tfidf_title.vocabulary_.keys())
)

def build_index():
    # считывает сырые данные и строит индекс
    for idx in range(len(documents)):
        index.append(Document(documents.loc[idx]))

def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее

    vector_body = tfidf_body.transform(query).toarray().squeeze()
    vector_title = tfidf_title.transform(query).toarray().squeeze()
    return random.random()

def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    words = preprocess.preprocessing_text(query)
    candidates = []
    for word in words:
        candidates = candidates & index[word] if not candidates else index[word]

    return candidates[:50]
