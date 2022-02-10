import re
from nltk.corpus import stopwords
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

def sentence_segmentation(text: str) -> list:
    if text and re.match(r'[\.!\?;]', text[-1]):
        text = text[:-1]
    return re.split(r'[\.!\?;]\s', text.lower())

def tokenization(sentences: list) -> list:
    return [re.split(r'[,:(\s\-)]*\s', s) for s in sentences]

def lemmatize_sent(tokens):
    result = []
    for word in tokens:
        result.append(morph.parse(word)[0].normal_form)
    return result

def lemmatization(sentences):
    return [lemmatize_sent(s) for s in sentences]

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
    