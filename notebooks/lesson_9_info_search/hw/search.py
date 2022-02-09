from collections import Counter
from collections.abc import Sequence
import pandas as pd
import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
import io
import numpy as np
from itertools import islice
import utils

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
        return set(sorted(self._index[w], key=lambda i: self.documents[i].count_words[w]))
    
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

    candidates_inersec = set()  # кандидаты при пересечение слов
    candidates_words = []    # кандидаты от каждого слова

    for word in words:
        docs = index[word]
        if len(candidates_inersec) == n:
            break

        if candidates_inersec:
            candidates_inersec &= docs
        else:
            candidates_inersec |= docs

        candidates_words.append(docs)

    cand_doc = list(islice(map(index.get_doc, candidates_inersec), n))
    candidates = list(candidates_inersec)[:n]

    if len(cand_doc) < n:
        word_n = int((n - len(cand_doc)) / (len(words) + 1)) # количество рекомендаций для каждого слова
        print(word_n)
        for i in range(len(candidates_words)):
            candidates_words[i] -= candidates_inersec
            cand_doc += list(
                islice(map(index.get_doc, candidates_words[i]), word_n), 
            )
            candidates += list(candidates_words[i])[:word_n]
    
    # metrics(candidates, cand_doc, query)
    return cand_doc

def metrics(candidates: list, cand_doc: list, query: str):
    """
    candidates неотсортированы, номера документов
    cand_doc неотсортированы, сами документы
    """
    if candidates:
        scores = np.array([score(query, doc) for doc in cand_doc])
        candidates = np.array(candidates)[np.argsort(scores)][::-1]
        actual = np.array([3216, 8964, 5635, 8584, 4371, 5753, 9313, 6940, 5585, 3453, 1234, 1325, 5634, 3454, 312, 34,6556])
        scores = np.sort(scores)[::-1]
        k = 10
        
        print(list(scores))
        print(f"For query: {query}")
        print(f"K = {k}")

        print("<----Average precision at K---->")
        print(f"AP@k = {utils.apk(actual, candidates, k)}")

        print("<----Discounted cumulative gain at K---->")
        print(f"DCG@k = {utils.dcgk(scores, k)}")

        print("<----PFound---->")
        print(f"PFound@k = {utils.pfound(scores, k)}")

def all_metrics():
    scores = np.array([
        [0.7715341140103038, 0.7107194371383811, 0.6441151173832724, 0.5805982351628356, 0.538931822423592, 0.530862106266415, 0.5298731333959161, 0.5291877913327061, 0.5284848432025107, 0.5227276443085094, 0.5209667690081797, 0.5132334480456557, 0.511633112309639, 0.5060679350477159, 0.5053212138011789, 0.4668336572456905, 0.46653162334390486, 0.46577259922700887, 0.4654904783811295, 0.4597036042834014, 0.458000019362805, 0.44997705630389406, 0.4444808798261719, 0.43645889656630077, 0.42617813073091065, 0.42537195897621, 0.4248069186806597, 0.41551034520593744, 0.4070354777878115, 0.4048046018976372, 0.4024532741214424, 0.3903873406150159, 0.388927681406217, 0.3865027007264734, 0.3800685356290967, 0.3525155088103029, 0.33559005472872727, 0.28150120797598704],
        [0.7752003339601666, 0.7488806263411703, 0.7279478618187782, 0.7242418088133045, 0.7191745928127145, 0.638172071221864, 0.6215147127213216, 0.6164291973266014, 0.614617801331914, 0.6141350511818342, 0.6092727245178737, 0.598699670459371, 0.5845667513194025, 0.5812579474611701, 0.5770044271999697, 0.5768080827067812, 0.5738270799802044, 0.5684940451604689, 0.5663246052119046, 0.5632504713538649, 0.5519281427811962, 0.5516856876326584, 0.5465149294725368, 0.5448692841124956, 0.5300370670907433, 0.5286228858432909, 0.5261632005139838, 0.5120030699239122, 0.4938756514763988, 0.4877404802937439, 0.46994493768825496, 0.4588592053829797, 0.4544515343840818, 0.4458497288991809],
        [0.8363025986175763, 0.6198793329945855, 0.6084170007088161, 0.5763989109095154, 0.5761570557367104, 0.5749883597863304, 0.574138537617396, 0.5465194286138466, 0.5452531195149384, 0.5215212543928949, 0.49932065240341994, 0.4977356376875082, 0.49669304511435375, 0.49522530454487557, 0.4930116790689797, 0.4898447657238951, 0.4837953296968006, 0.48343400787981494, 0.48255344912786047, 0.4817758641517365, 0.48088756668628985, 0.47954236206429235, 0.4508531029206805, 0.45034999780051654, 0.4443903063830811, 0.4370766756851304, 0.43344020866660493, 0.43314856859944817, 0.4304340808472442, 0.4272547352158848, 0.4271917262718701, 0.4270815996113597, 0.41280500912320517, 0.41261629381004317, 0.4085745326935294, 0.40640503713495657, 0.3988890225090109, 0.3968675327846718, 0.39338573215175765, 0.3579803613225495, 0.35727684466452525, 0.3561908290635397, 0.35200423429653666, 0.34549687998642586],
        [0.6949970118175788, 0.66799387692142, 0.625827781183635, 0.5879450495732774, 0.5732832404887174, 0.5529252965465568, 0.5523204184741559, 0.5424492520195704, 0.5401454285946212, 0.5109875745305889, 0.5109875745305889, 0.5096404085993077, 0.5096404085993077, 0.5078989404697393, 0.5051561037298848, 0.50428515264066, 0.5019476925251064, 0.4816390005680428, 0.48145064164979806, 0.47995273516382464, 0.47951793897781037, 0.4624698706122343, 0.45911627900992613, 0.4528660556063506, 0.4528660556063506, 0.4488648602872868, 0.4488365181109577, 0.4484225401436843, 0.4428372650332805, 0.44154264246698227, 0.4264039399233496, 0.4144529177198809, 0.38002576441478, 0.38002576441478, 0.3648897234188644, 0.32951086703433746, 0.3279268285842242],
        [0.8252094551569183, 0.6773762526662903, 0.6530906685742194, 0.6519850554432018, 0.5915022512389473, 0.5872211017117752, 0.5584579901185512, 0.5549922019843077, 0.5374323519257034, 0.5344722703251825, 0.532221511281701, 0.5316604115472803, 0.5300913577320929, 0.5288902171271853, 0.5218603854958324, 0.5218603854958324, 0.5126522783551302, 0.5119825504495233, 0.502594826526592, 0.5010205436788289, 0.49507397146948595, 0.4943306026970282, 0.48679720696324247, 0.4761642706431292, 0.4722825564863565, 0.4693458014343676, 0.4684611526268364, 0.4619118859017419, 0.45242435145644794, 0.4494386289824181, 0.4464620728628584, 0.4317742192098394, 0.4212246680108524, 0.4184295215586603, 0.41157066158335864, 0.40889198103130986, 0.40654244725300653, 0.3486645521580808]
    ])
    actual = np.array([
        [1810, 2821, 5664, 3210, 3361, 3361, 1039, 4403, 4483, 4864, 6600, 3453, 1234, 1325, 5634, 3454, 312, 34,6556],
        [1810, 2821, 5664, 3210, 3361, 3361, 4068, 20, 8871, 6285, 5612, 6920, 3830,5722, 1039, 4403, 4483, 4864, 6600, 3453, 1234, 1325, 5634, 3454, 312, 34,6556],
        [3216, 8964, 5635, 8584, 4371, 5753, 9313, 6940, 5585, 3453, 1234, 1325, 5634, 3454, 312, 34,6556],
        [6916, 849, 671,  2053, 2070, 8221, 2321, 3445, 2324, 1325, 5634, 3454, 312, 34,6556],
        [1810, 740, 6273, 6412, 5640, 2063, 5382, 4483, 4864, 6600, 3453, 1234, 1325, 5634, 3454, 312, 34,6556]
    ])
    predicted = np.array([
        [1810, 2821, 5664, 3210, 8064, 1690, 6916, 273, 6310, 2067, 4106, 6673, 7710, 5004, 6039, 4123, 9108, 932, 6802, 4098, 8467, 544, 2049, 382, 6167, 1, 2073, 8202, 540, 6161, 4103, 1283, 7704, 4101, 6426, 2055, 4230, 8220],
        [5635, 4114, 3085, 5863, 3972, 163, 7983, 3361, 20, 4068, 3623, 3624, 8871, 4102, 6285, 5612, 6920, 3830, 5140, 8204, 5722, 9218, 3587, 1039, 4403, 4483, 3245, 2262, 2063, 5154, 8202, 6600, 4864, 6692],
        [3216, 8964, 5635, 8584, 4371, 5753, 5863, 9313, 6940, 1799, 5722, 5585, 599, 6565, 5122, 2937, 9315, 5119, 5098, 5612, 3623, 8204, 9218, 5448, 472, 767, 3597, 3587, 7237, 163, 1025, 4102, 3039, 2658, 3788, 3329, 2679, 7334, 3223, 7855, 2278, 9028, 8202, 9228],
        [6916, 2053, 849, 2051, 671, 2070, 8221, 8206, 4111, 3, 3, 9, 9, 6157, 8232, 4144, 4107, 4115, 8204, 2067, 20, 6999, 3923, 8202, 8202, 27, 4126, 6146, 7907, 2161, 13, 16, 24, 24, 8199, 6184, 8215],
        [740, 6273, 6412, 5640, 5668, 2063, 5382, 4275, 479, 2067, 6665, 2053, 2052, 6160, 17, 17, 8077, 5783, 4226, 0, 4615, 8217, 3089, 1550, 518, 523, 18, 8195, 8192, 2101, 6145, 4111, 8206, 8584, 6193, 4624, 6164, 8198]
    ])
    k = 10

    print("<----Mean average precision at k---->")
    print(f"MAP@K = {utils.mapk(actual, predicted, k)}")

    print("<----Mean reciprocal rank (MRR) at k---->")
    print(f"MRR@k = {utils.mrr(actual, predicted, k)}")

if __name__ == '__main__': 
    all_metrics()