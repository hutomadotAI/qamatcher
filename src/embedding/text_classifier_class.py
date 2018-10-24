import dill
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

import logging
import logging.config


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


class EmbeddingComparison:
    @property
    def logger(self):
        return self.__logger

    def __init__(self, sen_emb=None, y=None, embedding_dim=400, random_seed=3435):
        self.__logger = _get_logger()
        self.random_seed = random_seed
        self.sen_emb = np.array(sen_emb) if sen_emb is not None else None
        self.sen_emb_red = None
        self.embedding_dim = embedding_dim
        self.y = y
        self.classes = list(set(y)) if y is not None else None
        self.pca = PCA(n_components=1, random_state=self.random_seed)

    @staticmethod
    def downscale_probas(probas):
        k = 0.3  # 0.2
        return k * probas / (k - probas + 1.)

    def fit(self, X, y):
        # subtracting 1st principal component according to
        # 'A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS' by Arora et. al.
        self.pca.fit(self.sen_emb)
        self.sen_emb_red = self.sen_emb - self.pca.components_[0]

    def predict(self, q_emb, subset_idx=None):
        if subset_idx:
            train_x = self.sen_emb[np.array(subset_idx), :]
            train_y = self.y[np.array(subset_idx)]
        else:
            train_x = self.sen_emb
            train_y = self.y
        # q_emb_red = q_emb - self.pca.components_[0]

        # compute cosine similarity
        cossim = np.dot(q_emb, train_x.T) / (np.outer(
            np.linalg.norm(q_emb, axis=1),
            np.linalg.norm(train_x, axis=1)))
        cossim = np.where(cossim < 0., 0., cossim)

        # most similar vector is the predicted class
        preds = np.argmax(cossim, 1)
        preds = [train_y[i] for i in preds]
        # probs = self.downscale_probas(np.max(cossim, axis=1))
        probs = np.max(cossim, axis=1)
        return preds, list(probs)

    def save_model(self, file_path: Path):
        self.__logger.debug("Saving model to {}".format(file_path))

        with file_path.open('wb') as f:
            dill.dump([
                self.sen_emb, self.sen_emb_red, self.y, self.pca, self.classes
            ], f)

    def load_model(self, file_path: Path):
        self.__logger.debug("Loading model from {}".format(file_path))
        with file_path.open('rb') as f:
            m = dill.load(f)

        if len(m) != 5:
            error_msg = "pkl file of saved model has wrong set of parameters;"\
               "len is {} - should be 5".format(len(m))
            self.__logger.error(error_msg)
            raise ModelError(error_msg)
        self.sen_emb = m[0]
        self.sen_emb_red = m[1]
        self.y = m[2]
        self.pca = m[3]
        self.classes = m[4]

    def get_classes(self):
        return self.y
