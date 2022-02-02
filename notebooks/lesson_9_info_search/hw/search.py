import random
from collections import Counter
from collections.abc import Sequence
import pickle
import pandas as pd
import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
import io
import numpy as np
from itertools import islice


documents = pd.read_pickle('data')

class Document:
    def __init__(self, doc):
        self.url = doc['issue_url'][1:-1]
        self.title = doc['issue_title']
        self.text = doc['body']
        self.vector_title = doc['vectors_title']
        self.vector_body = doc['vectors_body']

        self.count_words = Counter(doc['words_body'] + doc['words_title'])

    def format(self, query):
        words = preprocess.preprocessing_text(query)
        body = ""
        # for word in words:
        #     idx = self.text.find(word)
        #     if idx != -1:
        #         body += self.text[idx-50:idx+50] + " ..."
    
        body = body if body != "" else self.text[:200]
        return [self.title, body, self.url]
    
    def __equal__(self, other):
        return self.url == other.url
    
    def __ne__(self, other):
        return not self == other

    def count_word(self, word):
        return self.count_words[word]
    
    def __str__(self):
        return self.url


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
        return sorted(self._index[w], key=lambda i: self.documents[i].count_words[w])
    
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


def get_tfidf(type_tfidf):
    dummy_fun = lambda doc: doc
    tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)

    if type_tfidf == 'body':
        tfidf.fit(documents.words_body)
    elif type_tfidf == 'title':
        tfidf.fit(documents.words_title)

    return tfidf

def load_vectors(fname, limit):
  fin = io.open(fname, 'r', encoding = 'utf-8', newline = '\n', errors = 'ignore')
  n, d = map(int, fin.readline().split())
  data = {}
  for line in islice(fin, limit):
    tokens = line.rstrip().split(' ')
    data[tokens[0]] = np.array(list(map(float, tokens[1:])))
  return data

tfidf_body = get_tfidf('body')
tfidf_title = get_tfidf('title') 
index = Index(
    list(tfidf_body.vocabulary_.keys()) + list(tfidf_title.vocabulary_.keys())
)
vecs = load_vectors('crawl-300d-2M.vec', 200_000)

def get_vocab(tfidf):
    dim = 300
    zero = sum(vecs.values()) / len(vecs)

    vocab = np.zeros((len(tfidf.vocabulary_.keys()), dim))
    for key in tfidf.vocabulary_.keys():
        vocab[tfidf.vocabulary_[key]] = vecs.get(key, zero)
    return vocab

vocab_body = get_vocab(tfidf_body)
vocab_title = get_vocab(tfidf_title)

def build_index():
    for idx in range(len(documents)):
        index.append(Document(documents.loc[idx]))

def score(query, document):
    words = preprocess.preprocessing_text(query)

    tfidf_b = tfidf_body.transform([words]).toarray().squeeze()
    tfidf_t = tfidf_title.transform([words]).toarray().squeeze()

    vector_body, vector_title = tfidf_b.dot(vocab_body), tfidf_t.dot(vocab_title)

    diff_body = cosine_similarity([vector_body], [document.vector_body])[0][0]
    diff_title = cosine_similarity([vector_title], [document.vector_title])[0][0]

    return 0.4 * diff_body + 0.6 * diff_title

def retrieve(query):
    n = 50                      # количество документов на выдаче
    words = preprocess.preprocessing_text(query)
    word_n = int(n / (len(words) + 1)) # количество рекомендаций для каждого слов

    candidates_inersec = set()  # кандидаты при пересечение слов
    candidates_words = set()    # кандидаты от каждого слова

    for word in words:
        docs = index[word]
        if len(candidates_inersec) == n:
            break

        if candidates_inersec:
            candidates_inersec &= set(docs) 
        else:
            candidates_inersec = set(docs)

        candidates_words |= set(docs[:word_n])

    # чтобы объединить и первым поставить результат пересечений, удалим
    # из кандидатов всех слов пересечения
    candidates_words -= candidates_inersec

    cand_doc = list(islice(map(index.get_doc, candidates_inersec), n))
    if len(cand_doc) < n:
        cand_doc += list(
            islice(map(index.get_doc, candidates_words), n-len(cand_doc)), 
        )

    return cand_doc
