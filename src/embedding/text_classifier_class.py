# -*- coding: utf-8 -*-

import dill
import numpy as np
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

import logging
import logging.config

# from embedding.investigate import tsne_plot


def _get_logger():
    logger = logging.getLogger('embedding')
    return logger


class ModelError(Exception):
    """Model error"""
    pass


"""
This script defines a support-vector classifier using a
sentence embedding vector as input. The sentence embedding is
computed using word embedding vectors extracted from word2vec
which are weighted by their tf-idf weight.
"""


class TfidfEmbeddingVectorizer:
    def __init__(self, word2vec, dim, voc=None):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = dim  # len(word2vec.itervalues().next())
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


class EmbeddingComparison:
    @property
    def logger(self):
        return self.__logger

    def __init__(self, w2v=None, embeddingDim=300, random_seed=3435):
        self.__logger = _get_logger()
        self.random_seed = random_seed
        self.w2v = w2v if w2v is not None else {}
        self.embeddingDim = embeddingDim
        self.vectorizer = TfidfEmbeddingVectorizer(self.w2v, self.embeddingDim)
        self.pca = PCA(n_components=1, random_state=self.random_seed)
        self.X_tfidf = None
        self.y = None
        self.classes = None

    def scale_probas(self, probas):
        # positive part of tunable 'sigmoid' fct found at
        # https://dinodini.wordpress.com/2010/04/05/normalized-tunable-sigmoid-functions/
        return -10.5 * probas / (-probas - 9.5)

    def downscale_probas(self, probas):
        k = 0.2
        return k * probas / (k - probas + 1.)

    def update_w2v(self, dic):
        self.w2v.update(dic)
        self.vectorizer.update_w2v(dic)

    def get_unknown_words(self, words):
        return [w for w in words if w not in self.w2v.keys()]

    def fit(self, X, y):
        self.vectorizer.fit(X, y)
        self.X_tfidf = self.vectorizer.transform(X)
        # subtracting 1st principal component according to
        # 'A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS' by Arora et. al.
        self.pca.fit(self.X_tfidf)
        self.X_tfidf = self.X_tfidf - self.pca.components_[0]
        self.y = np.array(y)
        self.X = X
        self.classes = list(set(y))

    def predict(self, X, scale_probas=False, subset_idx=None):
        if subset_idx:
            train_x = self.X_tfidf[subset_idx]
            train_y = self.y[subset_idx]
        else:
            train_x = self.X_tfidf
            train_y = self.y
        target_tfidf = self.vectorizer.transform(X)
        target_tfidf = target_tfidf - self.pca.components_[0]
        # compute cosine similarity
        cossim = np.dot(target_tfidf, train_x.T) / (
            np.outer(np.linalg.norm(target_tfidf, axis=1), np.linalg.norm(train_x, axis=1)))
        # self.logger.info("cossim: {}".format(cossim))
        cossim = np.where(cossim < 0., 0., cossim)
        if subset_idx:
            self.logger.info("cossims: {}".format(cossim))
        # most similar vector is the predicted class
        preds = np.argmax(cossim, 1)
        preds = [train_y[i] for i in preds]
        probs = self.downscale_probas(np.max(cossim, axis=1))
        return preds, list(probs)

    def save_model(self, file_path: Path):
        self.__logger.debug("Saving model to {}".format(file_path))

        with file_path.open('wb') as f:
            dill.dump([self.X_tfidf, self.y, self.pca, self.classes,
                       self.vectorizer.word2weight, self.w2v], f)  # , self.X

    def load_model(self, file_path: Path):
        self.__logger.debug("Loading model from {}".format(file_path))
        with file_path.open('rb') as f:
            m = dill.load(f)

        if (len(m) != 6):
            error_msg = "pkl file of saved model has wrong set of parameters;"\
               "len is {} - should be 6".format(len(m))
            self.__logger.error(error_msg)
            raise ModelError(error_msg)
        self.vectorizer.word2weight = m[4]
        self.X_tfidf = m[0]
        self.y = m[1]
        self.pca = m[2]
        self.classes = m[3]
        self.update_w2v(m[5])
        # self.X = m[6]

    def get_classes(self):
        return self.y
