"""SVCLASSIFIER training worker processes"""
import logging
import datetime
import shutil
import tempfile
from pathlib import Path
import re
import aiohttp

import ai_training as ait
import ai_training.training_process as aitp

from embedding.text_classifier_class import EmbeddingComparison

from embedding.embedding_client import EmbeddingClient
from embedding.entity_wrapper import EntityWrapper
from embedding.svc_config import SvcConfig
from embedding.string_match import StringMatch

MODEL_FILE = "model.pkl"
DATA_FILE = "data.pkl"
TRAIN_FILE = "train.pkl"


def _get_logger():
    logger = logging.getLogger('embedding.training')
    return logger


class TrainEmbedMessage(aitp.TrainingMessage):
    """Message class for training a QA-Matcher"""

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
        self.entity_wrapper = EntityWrapper(config.er_server_url,
                                            self.aiohttp_client)
        self.embedding_client = EmbeddingClient(config.emb_server_url,
                                                self.aiohttp_client)
        self.string_match = StringMatch(self.entity_wrapper)
        self.last_update_sent = None
        self.callback = None
        self.regex_finder = re.compile(r'@{(.*?)}@')

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
            temp_train_file = tempdir_path / TRAIN_FILE

            self.logger.info("Extracting entities...")
            q_entities, a_entities = [], []
            for question, answer in q_and_a:
                q_entity = await self.entity_wrapper.extract_entities(question)
                a_entity = await self.entity_wrapper.extract_entities(answer)
                q_entities.append(q_entity)
                a_entities.append(a_entity)
                self.report_progress(0.1)
            self.entity_wrapper.save_data(temp_data_file, q_entities, a_entities, y)
            self.report_progress(0.2)

            self.logger.info(
                "Entities saved to {}, tokenizing...".format(temp_data_file))

            x_tokens_string_matcher = []
            x_tokens_string_matcher_no_sw = []
            x_cust_entities = []
            sen_emb = []
            for question in x:
                # find custom entities
                train_sample_ents = self.regex_finder.findall(question)
                x_cust_entities.append(train_sample_ents)

                # tokenize for string matcher
                tokens = await self.entity_wrapper.tokenize(question,
                                                            sw_size='large',
                                                            filter_ents='False')
                tokens_no_sw = await self.entity_wrapper.tokenize(question,
                                                                  sw_size='small',
                                                                  filter_ents='False')
                x_tokens_string_matcher.append(tokens)
                x_tokens_string_matcher_no_sw.append(tokens_no_sw)
                self.report_progress(0.3)

                # generate sentence embeddings
                q_emb = self.embedding_client.get_sentence_embedding(question)
                sen_emb.append(q_emb)
            self.report_progress(0.4)

            self.string_match.save_train_data([q_and_a, x_tokens_string_matcher, x_cust_entities,
                                               x_tokens_string_matcher_no_sw],
                                              temp_train_file)

            cls = EmbeddingComparison(sen_emb=sen_emb, y=y)
            # cls.fit()
            self.report_progress(0.8)

            cls.save_model(temp_model_file)
            self.logger.info("Saved model to {}".format(temp_model_file))
            self.report_progress(0.9)

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
