"""SVCLASSIFIER training worker processes"""
import logging
import datetime
import shutil
import tempfile
from pathlib import Path

import aiohttp

import ai_training as ait
import ai_training.training_process as aitp

from embedding.text_classifier_class import EmbeddingComparison

from embedding.word2vec_client import Word2VecClient
from embedding.entity_wrapper import EntityWrapper
from embedding.svc_config import SvcConfig

MODEL_FILE = "model.pkl"
DATA_FILE = "data.pkl"


def _get_logger():
    logger = logging.getLogger('embedding.training')
    return logger


class TrainEmbedMessage(aitp.TrainingMessage):
    """Message class for training a SVCLASSIFIER"""

    def __init__(self, ai_path, ai_id, max_training_mins: int):
        super().__init__(ai_path, ai_id, max_training_mins)


UPDATE_EVERY_N_SECONDS = 10.0


class EmbedTrainingProcessWorker(aitp.TrainingProcessWorkerABC):
    def __init__(self, pool, asyncio_loop, aiohttp_client_session=None):
        super().__init__(pool, asyncio_loop)
        self.callback_object = None
        self.logger = _get_logger()
        if aiohttp_client_session is None:
            aiohttp_client_session = aiohttp.ClientSession()

        self.aiohttp_client = aiohttp_client_session
        config = SvcConfig.get_instance()
        self.w2v_client = Word2VecClient(config.w2v_server_url,
                                         self.aiohttp_client)
        self.entity_wrapper = EntityWrapper(config.er_server_url,
                                            self.aiohttp_client)
        self.last_update_sent = None
        self.callback = None

    async def get_vectors(self, questions):
        word_dict = {}
        for question in questions:
            words = question.split(' ')
            for word in words:
                word_dict[word] = ""
        return await self.w2v_client.get_vectors_for_words(word_dict)

    def report_progress(self, progress_value):
        now = datetime.datetime.utcnow()
        seconds_since_update = (now - self.last_update_sent).total_seconds() \
            if self.last_update_sent is not None \
            else (UPDATE_EVERY_N_SECONDS + 1.0)
        if seconds_since_update > UPDATE_EVERY_N_SECONDS:
            if self.callback is not None:
                self.callback.report_progress(progress_value)
                self.callback.check_for_cancel()
            self.last_update_sent = now

    async def train(self, msg, topic: ait.Topic, callback):
        # handshake with API to say we're starting training
        self.callback = callback
        self.last_update_sent = None
        if callback is not None:
            await callback.wait_to_save()

        training_file_path = msg.ai_path / ait.AI_TRAINING_STANDARD_FILE_NAME
        self.logger.info(
            "Start training using file {}".format(training_file_path))
        root_topic = ait.file_load_training_data_v1(training_file_path)

        q_and_a = [(entry.question, entry.answer)
                   for entry in root_topic.entries]
        x, y = zip(*q_and_a)

        # save to a temp directory first (self-cleaning)
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)
            temp_model_file = tempdir_path / MODEL_FILE
            temp_data_file = tempdir_path / DATA_FILE

            self.logger.info("Extracting entities...")
            entities = []
            for question in x:
                entity = await self.entity_wrapper.extract_entities(question)
                entities.append(entity)
                self.report_progress(0.1)
            self.entity_wrapper.save_data(temp_data_file, entities, y)
            self.report_progress(0.2)

            self.logger.info(
                "Entities saved to {}, tokenizing...".format(temp_data_file))

            x_tokens = []
            for question in x:
                tokens = await self.entity_wrapper.tokenize(question)
                x_tokens.append(tokens)
                self.logger.info("tokens: {}".format((question, tokens)))
                self.report_progress(0.3)

            x_tokens_set = list(set([w for l in x_tokens for w in l]))
            self.report_progress(0.4)

            words = {}
            for l in x_tokens_set:
                words[l] = None

            vecs = await self.get_vectors(list(words.keys()))
            self.report_progress(0.6)

            cls = EmbeddingComparison(w2v=vecs)
            self.logger.info("Fitting...")
            cls.fit(x_tokens, y)
            self.report_progress(0.8)

            cls.save_model(temp_model_file)
            self.logger.info("Saved model to {}".format(temp_model_file))
            self.report_progress(0.9)

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
