"""SVCLASSIFIER training worker processes"""
import logging
import itertools
import os
import datetime
import aiohttp

import ai_training as ait
import ai_training.training_process as aitp
import ai_training.training_file as aitp_tfile

from text_classifier_class import EmbeddingComparison
from spacy_wrapper import SpacyWrapper

from word2vec_client import Word2VecClient
from svc_config import SvcConfig

MODEL_FILE = "model.pkl"


def _get_logger():
    logger = logging.getLogger('embedding.training')
    return logger


class TrainEmbedMessage(aitp.TrainingMessage):
    """Message class for training a SVCLASSIFIER"""

    def __init__(self, ai_path, ai_id, max_training_mins: int):
        super().__init__(ai_path, ai_id, max_training_mins)


class EmbedTrainingProcessWorker(aitp.TrainingProcessWorkerABC):
    def __init__(self, pool, asyncio_loop):
        super().__init__(pool, asyncio_loop)
        self.callback_object = None
        self.logger = _get_logger()
        self.w2v_client = Word2VecClient(
            SvcConfig.get_instance().w2v_server_url)

    async def get_vectors(self, questions):
        word_dict = {}
        for question in questions:
            words = question.split(' ')
            for word in words:
                word_dict[word] = ""
        return await self.w2v_client.get_vectors_for_words(word_dict)

    async def train(self, msg, topic: ait.Topic, callback_object):

        training_file = os.path.join(
            str(msg.ai_path), aitp_tfile.AI_TRAINING_STANDARD_FILE_NAME)
        model_file = os.path.join(str(msg.ai_path), MODEL_FILE)
        self.logger.info("Start training using file {}".format(training_file))
        x, y = self.load_train_data(training_file)

        spacy_wrapper = SpacyWrapper()
        self.logger.info("Tokenizing...")
        x_tokens = [spacy_wrapper.tokenizeSpacy(s) for s in x]
        x_tokens_set = list(set([w for l in x_tokens for w in l]))

        words = {}
        for l in x_tokens_set:
            words[l] = None

        try:
            vecs = await self.get_vectors(list(words.keys()))
        except aiohttp.client_exceptions.ClientConnectorError as exc:
            self.logger.error(
                "Could not receive response from w2v service - {}".format(exc))
            return ait.AiTrainingState.ai_error, None

        cls = EmbeddingComparison(w2v=vecs)
        self.logger.info("Fitting...")
        cls.fit(x_tokens, y)
        saved_model_file = cls.save_model(model_file)
        self.logger.info("Saved model to {}".format(saved_model_file))

        now = datetime.datetime.now()
        hash_value = now.strftime("%y%m%d.%H%M%S")
        training_data_hash = hash_value
        result = (ait.AiTrainingState.ai_training_complete, training_data_hash)
        return result

    def grouper(self, iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    def to_utf8(self, string):
        if type(string) == bytes:
            return string.decode('utf-8', 'ignore')
        else:
            return string

    def load_train_data(self, file_name=None):
        self.logger.info("loading data from {}".format(file_name))
        nLines = 3
        y = []
        X = []
        isQuestion = True
        with open(file_name, 'r') as f:
            for lines in self.grouper(f, nLines, ''):
                for line in lines:
                    line = line.strip()
                    if line == '\n' or len(line) == 0:
                        continue
                    elif not isQuestion:
                        y.append(self.to_utf8(line))
                        isQuestion = True
                    else:
                        X.append(self.to_utf8(line))
                        isQuestion = False

            # check if we have more questions than answers
            if len(X) > len (y):
                # remove last question
                self.logger.warn("Last question removed as it didn't have a corresponding answer.")
                X = X[:-1]
        return X, y
