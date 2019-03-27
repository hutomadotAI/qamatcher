"""SVCLASSIFIER training worker processes"""
import logging
import datetime
import shutil
import tempfile
from pathlib import Path
import re

import ai_training as ait

from embedding.training_process_base import EmbedTrainingProcessWorker
from embedding.word2vec_client import Word2VecClient
from embedding.entity_wrapper import EntityWrapperPlus
from embedding.string_match import StringMatch

MODEL_FILE = "model.pkl"
DATA_FILE = "data.pkl"
TRAIN_FILE = "train.pkl"


def _get_logger():
    logger = logging.getLogger('qa_matcher.training')
    return logger


UPDATE_EVERY_N_SECONDS = 10.0


class QAMatcherTrainingProcessWorker(EmbedTrainingProcessWorker):
    def __init__(self, pool, aiohttp_client_session=None):
        super().__init__(pool, aiohttp_client_session)
        self.logger = _get_logger()
        self.w2v_client = Word2VecClient(self.config.w2v_server_url,
                                         self.aiohttp_client)
        self.entity_wrapper = EntityWrapperPlus(self.config.er_server_url,
                                                self.aiohttp_client)
        self.string_match = StringMatch(self.entity_wrapper)
        self.regex_finder = re.compile(r'@{(.*?)}@')

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

    async def extract_entities(self, q_and_a):
        q_entities, a_entities = [], []
        for question, answer in q_and_a:
            q_entity = await self.entity_wrapper.extract_entities(question)
            a_entity = await self.entity_wrapper.extract_entities(answer)
            q_entities.append(q_entity)
            a_entities.append(a_entity)
        return q_entities, a_entities

    def extract_custom_entities(self, questions):
        ents = [self.regex_finder.findall(question) for question in questions]
        return ents

    async def setup_string_matcher(self, q_and_a, temp_train_file):
        x, y = zip(*q_and_a)

        # find custom entities for string matcher
        x_cust_entities = self.extract_custom_entities(x)

        # tokenize for string matcher
        x_tokens_string_matcher = await self.tokenize(
            x, sw_size='large', filter_ents='False')
        x_tokens_string_matcher_no_sw = await self.tokenize(
            x, sw_size='small', filter_ents='False')

        # save data to temp file
        self.string_match.save_train_data([q_and_a, x_tokens_string_matcher, x_cust_entities,
                                           x_tokens_string_matcher_no_sw],
                                          temp_train_file)

    async def setup_entity_matcher(self, q_and_a, temp_train_file):
        x, y = zip(*q_and_a)

        # extract entities for entity matcher
        q_entities, a_entities = await self.extract_entities(q_and_a)
        self.entity_wrapper.save_data(temp_train_file, q_entities, a_entities, y)
        self.logger.info(
            "Entities saved to {}, tokenizing...".format(temp_train_file))

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

        # save to a temp directory first (self-cleaning)
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)
            temp_model_file = tempdir_path / MODEL_FILE
            temp_data_file = tempdir_path / DATA_FILE
            temp_train_file = tempdir_path / TRAIN_FILE

            # prepare data for entity matcher
            self.logger.info("Extracting entities...")
            await self.setup_entity_matcher(q_and_a, temp_data_file)
            self.report_progress(0.3)
            self.logger.info("data preparation finished for entity matcher")

            # prepare data for string matcher
            await self.setup_string_matcher(q_and_a, temp_train_file)
            self.report_progress(0.6)
            self.logger.info("data preparation finished for string matcher")

            # prepare data for embedding matcher
            await self.setup_embedding_matcher(q_and_a, temp_model_file)
            self.report_progress(0.9)
            self.logger.info("data preparation finished for embedding matcher")

            self.logger.info("Moving training files to {}".format(msg.ai_path))
            model_file = msg.ai_path / MODEL_FILE
            data_file = msg.ai_path / DATA_FILE
            train_file = msg.ai_path / TRAIN_FILE
            shutil.move(str(temp_model_file), str(model_file))
            shutil.move(str(temp_data_file), str(data_file))
            shutil.move(str(temp_train_file), str(train_file))

        now = datetime.datetime.now()
        hash_value = now.strftime("%y%m%d.%H%M%S")
        training_data_hash = hash_value
        result = (ait.AiTrainingState.ai_training_complete, training_data_hash)
        return result
