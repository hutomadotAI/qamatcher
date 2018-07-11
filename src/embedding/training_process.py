"""SVCLASSIFIER training worker processes"""
import logging
import datetime
import shutil
import tempfile
from pathlib import Path

import aiohttp

import ai_training as ait
import ai_training.training_process as aitp


from text_classifier_class import EmbeddingComparison
from entity_matcher import EntityMatcher
from spacy_wrapper import SpacyWrapper

from word2vec_client import Word2VecClient
from svc_config import SvcConfig

MODEL_FILE = "model.pkl"
DATA_FILE = "data.pkl"


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
        self.spacy_wrapper = SpacyWrapper()

    async def get_vectors(self, questions):
        word_dict = {}
        for question in questions:
            words = question.split(' ')
            for word in words:
                word_dict[word] = ""
        return await self.w2v_client.get_vectors_for_words(word_dict)

    async def train(self, msg, topic: ait.Topic, callback_object):

        training_file_path = msg.ai_path / ait.AI_TRAINING_STANDARD_FILE_NAME
        self.logger.info("Start training using file {}".format(training_file_path))
        root_topic = ait.file_load_training_data_v1(training_file_path)

        q_and_a = [(entry.question, entry.answer) for entry in root_topic.entries]
        x, y = zip(*q_and_a)

        # save to a temp directory first (self-cleaning)
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)
            temp_model_file = tempdir_path / MODEL_FILE
            temp_data_file = tempdir_path / DATA_FILE

            self.logger.info("Extracting entities...")
            ent_matcher = EntityMatcher(spacy=self.spacy_wrapper)
            entities = [ent_matcher.extract_entities(s) for s in x]
            ent_matcher.save_data(temp_data_file, entities, y)

            self.logger.info("Entities saved to {}, tokenizing...".format(temp_data_file))
            x_tokens = [self.spacy_wrapper.tokenizeSpacy(s) for s in x]
            self.logger.info("tokens: {}".format([(xx, toks) for xx, toks in zip(x, x_tokens)]))
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
            cls.save_model(temp_model_file)
            self.logger.info("Saved model to {}".format(temp_model_file))

            self.logger.info("Moving training files to {}".format(msg.ai_path))
            model_file = msg.ai_path / MODEL_FILE
            data_file = msg.ai_path / DATA_FILE
            shutil.move(str(temp_model_file), str(model_file))
            shutil.move(str(temp_data_file), str(data_file))

        now = datetime.datetime.now()
        hash_value = now.strftime("%y%m%d.%H%M%S")
        training_data_hash = hash_value
        result = (ait.AiTrainingState.ai_training_complete, training_data_hash)
        return result
