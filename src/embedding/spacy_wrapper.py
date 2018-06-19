#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import spacy


class SpacyWrapper(object):

    parser = None

    # A custom stoplist
    STOPLIST = set(
        stopwords.words('english') + [u"n't", u"'s", u"'m", u"ca"] +
        list(ENGLISH_STOP_WORDS))
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
