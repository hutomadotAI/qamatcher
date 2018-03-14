# -*- coding: utf-8 -*-

import dill
import numpy as np
from collections import defaultdict

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import logging
import logging.config


def _get_logger():
    logger = logging.getLogger('embedding')
    return logger


"""
This script defines a support-vector classifier using a
sentence embedding vector as input. The sentence embedding is 
computed using word embedding vectors extracted from word2vec 
which are weighted by their tf-idf weight. 
"""


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim, voc=None):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = dim  #len(word2vec.itervalues().next())
        self.tfidf = None
        self.voc = voc

    def update_w2v(self, dic):
        self.word2vec.update(dic)

    def fit(self, X, y):
        self.tfidf = TfidfVectorizer(
            vocabulary=self.voc, decode_error='replace', analyzer=lambda x: x)
        self.tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(self.tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, [(w, self.tfidf.idf_[i])
                              for w, i in self.tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
            np.mean(
                [
                    self.word2vec[w] * self.word2weight[w] for w in words
                    if w in self.word2vec
                ] or [np.zeros(self.dim)],
                axis=0) for words in X
        ])


class EmbeddingComparison(object):

    @property
    def logger(self):
        return self.__logger

    def __init__(self, w2v, embeddingDim = 300):
        self.__logger = _get_logger()
        self.w2v = w2v
        self.embeddingDim = embeddingDim
        self.vectorizer = TfidfEmbeddingVectorizer(
            self.w2v, self.embeddingDim)
        self.pca = PCA(n_components=1)
        self.X_tfidf = None
        self.y = None
        self.classes = None

    def update_w2v(self, dic):
        self.w2v.update(dic)
        self.vectorizer.update_w2v(dic)

    def fit(self, X, y):
        self.vectorizer.fit(X, y)
        self.X_tfidf = self.vectorizer.transform(X)
        # subtracting 1st principal component according to
        # 'A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS' by Arora et. al.
        self.pca.fit(self.X_tfidf)
        self.X_tfidf = self.X_tfidf - self.pca.components_[0]
        self.y = np.array(y)
        self.classes = list(set(y))

    def predict(self, X):
        target_tfidf = self.vectorizer.transform(X)
        target_tfidf = target_tfidf - self.pca.components_[0]
        probs = {}
        # compute cosine similarity
        cossim = np.dot(target_tfidf, self.X_tfidf.T) / (
            np.outer(np.linalg.norm(target_tfidf, axis=1), np.linalg.norm(self.X_tfidf, axis=1)))
        # most similar vector is the predicted class
        preds = np.argmax(cossim, 1)
        preds = [self.y[i] for i in preds]

        # compute probas for classes
        cosmin = np.min(cossim, 1)
        tmp = cossim - cosmin.reshape([len(target_tfidf),1])
        prob = tmp / np.sum(tmp, 1).reshape([len(target_tfidf),1])
        prob_class = np.zeros((len(target_tfidf), len(self.classes)))
        for i, cl in enumerate(self.classes):
            idx = np.where(self.y == cl)[0]
            prob_class[:,i] = np.sum(prob[:,idx], 1)
        # sum of class vectors with largest prob is prediction
        # preds = np.argmax(prob_class, 1)
        # preds = [self.classes[i] for i in preds]

        # res = [(pred, prob) for pred, prob in zip(preds[:10], prob_class[:10,:])]
        # print(tabulate(res, headers=('pred','prob')))
        return preds, np.max(prob_class, 1).tolist()

    def save_model(self, file_path):
        self.__logger.debug("Saving model to {}".format(file_path))
        with open(file_path, 'wb') as f:
            dill.dump([self.X_tfidf, self.y, self.pca, self.classes,
                         self.vectorizer.word2weight], f)
        return file_path

    def load_model(self, file_path):
        self.__logger.debug("Loading model from {}".format(file_path))
        with open(file_path, 'rb') as f:
            m = dill.load(f)
        self.vectorizer.word2weight = m[4]
        self.X_tfidf = m[0]
        self.y = m[1]
        self.pca = m[2]
        self.classes = m[3]

    def get_classes(self):
        return self.y

