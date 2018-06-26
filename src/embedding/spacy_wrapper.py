#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import spacy


class SpacyWrapper(object):

    parser = None

    # A custom stoplist taken from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    CUSTOM_STOPLIST = ['much', 'herein', 'thru', 'per', 'somehow', 'throughout', 'almost', 'somewhere', 'whereafter',
                       'nevertheless', 'indeed', 'hereby', 'across', 'within', 'co', 'yet', 'elsewhere', 'whence',
                       'seeming', 'un', 'whither', 'mine', 'whether', 'also', 'thus', 'amongst', 'thereafter',
                       'mostly', 'amoungst', 'therefore', 'seems', 'something', 'thereby', 'others', 'hereupon', 'us',
                       'everyone', 'perhaps', 'please', 'hence', 'due', 'seemed', 'else', 'beside', 'therein',
                       'couldnt', 'moreover', 'anyway', 'whatever', 'anyhow', 'de', 'among', 'besides', 'though',
                       'either', 'rather', 'might', 'noone', 'eg', 'thereupon', 'may', 'namely', 'ie', 'sincere',
                       'whereby', 'con', 'latterly', 'becoming', 'meanwhile', 'afterwards', 'thence', 'whoever',
                       'otherwise', 'anything', 'however', 'whereas', 'although', 'hereafter', 'already', 'beforehand',
                       'etc', 'whenever', 'even', 'someone', 'whereupon', 'inc', 'sometimes', 'ltd', 'cant']
    STOPLIST = stopwords.words('english') + [u"n't", u"'s", u"'m", u"ca"] + CUSTOM_STOPLIST
    STOPLIST = set([s for s in STOPLIST
                    if s not in ['why', 'when', 'where', 'why', 'how', 'which', 'what', 'whose', 'whom']])
    
    # List of symbols we don't care about
    SYMBOLS = " ".join(string.punctuation).split(" ") + [
        u"-----", u"---", u"...", u"“", u"”", u'"', u"'ve"
    ]

    def __init__(self):
        self.logger = logging.getLogger('spacy.tokenizer')
        if SpacyWrapper.parser is None:
            SpacyWrapper.parser = spacy.load('en')

    def tokenizeSpacy(self, sample):
        # get the tokens using spaCy
        # self.logger.info("**** sample: {}".format(sample))
        tokens = SpacyWrapper.parser(sample)

        # lemmatize
        lemmas = []
        for tok in tokens:
            # don't lemmatize or lower case if word is all caps
            if tok.text.isupper():
                lemmas.append(tok.text)
            elif tok.lemma_ != "-PRON-":
                lemmas.append(tok.lemma_.lower().strip())
            else:
                lemmas.append(tok.lower_)

        tokens = lemmas

        # stoplist symbols
        tokens = [tok for tok in tokens if tok not in SpacyWrapper.SYMBOLS]

        # stoplist the tokens
        tmp = [tok for tok in tokens if tok not in SpacyWrapper.STOPLIST]
        if len(tmp) > 0:
            tokens = tmp

        # remove large strings of whitespace
        while "" in tokens:
            tokens.remove("")
        while " " in tokens:
            tokens.remove(" ")
        while "\n" in tokens:
            tokens.remove("\n")
        while "\n\n" in tokens:
            tokens.remove("\n\n")

        return tokens
