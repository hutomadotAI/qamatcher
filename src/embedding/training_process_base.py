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
from embedding.svc_config import SvcConfig

"""
This is the base class of the training process for Hutoma chat console
It generates weighted word vectors for each question in the training set
and compares them during a chat session using cosine-similarity.
It connects to an external word2vec service to collect the word embeddings
for the words in the phrases
"""

MODEL_FILE = "model.pkl"


def _get_logger():
    logger = logging.getLogger('embedding.training')
    return logger


class TrainEmbedMessage(aitp.TrainingMessage):
    """Message class for training a QA-Matcher"""

    def __init__(self, ai_path, ai_id, max_training_mins: int):
        super().__init__(ai_path, ai_id, max_training_mins)


UPDATE_EVERY_N_SECONDS = 10.0


class EmbedTrainingProcessWorker(aitp.TrainingProcessWorkerABC):
    def __init__(self, pool, aiohttp_client_session=None):
        super().__init__(pool)
        self.callback_object = None
        self.logger = _get_logger()
        if aiohttp_client_session is None:
            aiohttp_client_session = aiohttp.ClientSession()

        self.aiohttp_client = aiohttp_client_session
        self.config = SvcConfig.get_instance()
        self.w2v_client = Word2VecClient(self.config.w2v_server_url,
                                         self.aiohttp_client)
        self.last_update_sent = None
        self.callback = None

    async def tokenize(self, questions, sw_size, filter_ents):
        tokens = [await self.entity_wrapper.tokenize(
            question, sw_size=sw_size, filter_ents=filter_ents)
                  for question in questions]
        return tokens

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

    async def get_word_vectors(self, x_tokens):
        # find unknown words to word-embedding and get rid of them
        x_tokens_set = list(set([w for l in x_tokens for w in l]))
        unk_words = await self.w2v_client.get_unknown_words(x_tokens_set)
        self.logger.info("unknown words: {}".format(unk_words))
        x_tokens_set = [w for w in x_tokens_set if w not in unk_words]
        x_tokens = [[w for w in s if w not in unk_words] for s in x_tokens]

        # deal with empty sets/lists
        if not x_tokens_set:
            x_tokens_set = ['UNK']
        x_tokens = [l if len(l) > 0 else ['UNK'] for l in x_tokens]

        vecs = await self.get_vectors(x_tokens_set)
        return vecs, x_tokens

    async def setup_embedding_matcher(self, q_and_a, temp_train_file):
        x, y = zip(*q_and_a)

        # tokenize for embedding
        x_tokens = await self.tokenize(x, sw_size='xlarge', filter_ents='True')
        self.report_progress(0.7)

        # get word vectors to initialise embedding model
        vecs, x_tokens = await self.get_word_vectors(x_tokens)
        self.report_progress(0.75)
        for question, tokens in zip(x, x_tokens):
            self.logger.info("tokens: {}".format((question, tokens)))

        # initialise the embedding model
        cls = EmbeddingComparison(w2v=vecs)
        self.logger.info("Fitting...")
        cls.fit(x_tokens, y)
        self.report_progress(0.8)
        cls.save_model(temp_train_file)
        self.logger.info("Saved model to {}".format(temp_train_file))

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

            # prepare data for embedding matcher
            await self.setup_embedding_matcher(q_and_a, temp_model_file)
            self.report_progress(0.9)
            self.logger.info("data preparation finished for embedding matcher")

            self.logger.info("Moving training files to {}".format(msg.ai_path))
            model_file = msg.ai_path / MODEL_FILE
            shutil.move(str(temp_model_file), str(model_file))

        now = datetime.datetime.now()
        hash_value = now.strftime("%y%m%d.%H%M%S")
        training_data_hash = hash_value
        result = (ait.AiTrainingState.ai_training_complete, training_data_hash)
        return result
