#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import spacy


class SpacyWrapper(object):

    parser = None

    # A custom stoplist
    STOPLIST = set(
        stopwords.words('english') + [u"n't", u"'s", u"'m", u"ca"] +
        list(ENGLISH_STOP_WORDS))  #
    # List of symbols we don't care about
    SYMBOLS = " ".join(string.punctuation).split(" ") + [
        u"-----", u"---", u"...", u"“", u"”", u"'ve"
    ]

    def __init__(self):
        if SpacyWrapper.parser is None:
            SpacyWrapper.parser = spacy.load('en')

    def tokenizeSpacy(self, sample):
        # get the tokens using spaCy
        try:
            tokens = SpacyWrapper.parser(sample, "utf-8")
        except:
            tokens = SpacyWrapper.parser(sample)

        # lemmatize
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip()
                          if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas

        # tokens = [tok.orth_ for tok in tokens]

        # stoplist the tokens
        tokens = [tok for tok in tokens if tok not in SpacyWrapper.STOPLIST]

        # stoplist symbols
        tokens = [tok for tok in tokens if tok not in SpacyWrapper.SYMBOLS]

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
