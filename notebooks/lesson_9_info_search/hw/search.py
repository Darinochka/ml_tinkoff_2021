import random
from collections import Counter
from collections.abc import Sequence
import pickle
import pandas as pd
import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
from typing import Union


class Document:
    def __init__(self, doc):
        # можете здесь какие-нибудь свои поля подобавлять
        self.url = doc['issue_url']
        self.title = doc['issue_title']
        self.text = doc['body']
        self.vector_title = doc['vectors_title']
        self.vector_body = doc['vectors_body']

        self.count_words = Counter(doc['words_title'] + doc['words_title'])

    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text[:1000] + ' ...', self.url]
    
    def __equal__(self, other):
        return self.url == other.url
    
    def __ne__(self, other):
        return not self == other

    def count_word(self, word):
        return self.count_words[word]


class Index(Sequence):
    def __init__(self, vocab):
        self.documents = list()
        self._index = {word: set() for word in vocab}

    def append(self, doc: Document):
        self.documents.append(doc)
        self.update_index(doc)

    def update_index(self, doc):
        for word in doc.count_words:
            self._index[word].add(len(self.documents) - 1)

    def __getitem__(self, w: str) -> set:
        # return sorted(self.index[w], key=lambda x: self.documents[x].count_word(w))
        return self._index[w]
    
    def __len__(self):
        return len(self._index)
    
    def get_doc(self, idx):
        return self.documents[idx]

    def get_docs(self, idxs: Union[list, set]):
        docs = list()
        for idx in idxs:
            docs.append(self.get_doc(idx))
        return docs

    def get_index(self):
        return self._index
  
def get_data(file: str):
    df = pd.read_csv(file)

    df.words_body = df.words_body.apply(ast.literal_eval)
    df.words_title = df.words_title.apply(ast.literal_eval)
    df.vectors_title = df.vectors_title.apply(ast.literal_eval)
    df.vectors_body = df.vectors_body.apply(ast.literal_eval)

    return df
    
# documents = get_data('data.csv')
documents = pd.read_pickle('data')

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

    return random.random()

def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    words = preprocess.preprocessing_text(query)
    print(f"query words: {words}")

    candidates_inersec = set()  # кандидаты при пересечение слов
    candidate_words = set()     # кандидаты от каждого слова

    for word in words:
        if candidates_inersec:
            candidates_inersec &= index[word] 
        else:
            candidates_inersec = index[word]

        candidate_words |= index[word]

    # чтобы объединить и первым поставить результат пересечений, удалим
    # из кандидатов всех слов пересечения
    candidate_words -= candidates_inersec
    
    print(candidates_inersec, candidate_words)
    print(candidates_inersec.union(candidate_words))
    candidated_doc = index.get_docs(candidates_inersec.union(candidate_words))
    return candidated_doc[:50]
