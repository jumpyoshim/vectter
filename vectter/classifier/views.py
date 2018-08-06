import glob
import os

import MeCab
from gensim import corpora, matutils


def doc2word(doc: str) -> list:
    tagger = MeCab.Tagger('mecabrc')
    node = tagger.parseToNode(doc)
    words = []
    while node:
        meta = node.feature.split(',')
        if meta[0] == '名詞':
            words.append(node.surface.lower())
        node = node.next
    return words

def generate_dictionary(data: list) -> type:
    dictionary = corpora.Dictionary(data)
    if os.path.isfile('Dictionary.txt'):
        dictionary.load_from_text('Dictionary.txt')
    else:
        dictionary.save_as_text('Dictionary.txt')
    return dictionary

def curpus2dense(dictionary: type, doc: list) -> list:
    bow = dictionary.doc2bow(doc)
    dense = list(matutils.corpus2dense([bow], num_terms=len(dictionary)).T[0])
    return dense

if __name__ == '__main__':
    files = glob.glob('./data/*.txt')
    data = []
    for file in files:
        with open(file) as f:
            doc = f.read()
        data.append(doc2word(doc))
    dictionary = generate_dictionary(data)
    for doc in data:
        curpus2dense(dictionary, doc)
