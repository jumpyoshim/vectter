import MeCab
from gensim import corpora, matutils


def main(doc):
    tagger = MeCab.Tagger('mecabrc')
    node = tagger.parseToNode(doc)
    words = []
    while node:
        meta = node.feature.split(',')
        if meta[0] == '名詞':
            words.append(node.surface.lower())
        node = node.next

    print(words)
    dic = corpora.Dictionary(words)
    dic.filter_extremes(no_below=20, no_above=0.3)
    dic.save_as_text('Dictionary.txt')

    bow_corpus = dic.doc2bow(words)
    print(bow_corpus)
    dense = list(matutils.corpus2dense([bow_corpus], num_terms=len(dic)).T[0])
    print(dense)
