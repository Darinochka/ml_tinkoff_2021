import re
from nltk import  pos_tag
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


def sentence_segmentation(text: str) -> list:
    if text and re.match(r'[\.!\?;]', text[-1]):
        text = text[:-1]
    return re.split(r'[\.!\?;]\s', text.lower())

def tokenization(sentences: list) -> list:
    return [re.split(r'[,:(\s\-)]*\s', s) for s in sentences]

def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wn.ADJ,
        'V': wn.VERB,
        'N': wn.NOUN,
        'R': wn.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wn.NOUN

def lemmatization(sentences: list) -> list:
    sentences_tag  = [pos_tag(s) for s in sentences] # получаем теги слов каждого предложения
    
    lemmatizer = WordNetLemmatizer()
    lemm_sentences = []
    for sent in sentences_tag:
        pos_tagged = [(word, get_wordnet_pos(tag)) for word, tag in sent]
        lemm_sentences.append([lemmatizer.lemmatize(word, tag) for word, tag in pos_tagged])

    return lemm_sentences

def del_stopwords(sentences: list) -> list:
    stop_words = set(stopwords.words('english')).union({'', ' '})
    upd_sentences = []
    re_sub = lambda x: re.sub(r"[\+=\t\r\n,;:\*'\"]+","", x)
    union_sentences = lambda x: list(set().union(*x))

    for sent in sentences:
        upd_sentences.append([
            re_sub(word) for word in sent if re_sub(word) not in stop_words and len(word) not in [1, 2]
        ])
    
    return union_sentences(upd_sentences)

def preprocessing_text(text: str) -> list:
    return del_stopwords(lemmatization(tokenization(sentence_segmentation(text))))