"""SVCLASSIFIER chat worker processes"""

import logging
from pathlib import Path
import time
import aiohttp

import ai_training.chat_process as ait_c
from embedding.entity_wrapper_base import EntityWrapper
from embedding.text_classifier_class import EmbeddingComparison
from embedding.word2vec_client import Word2VecClient
from embedding.svc_config import SvcConfig


"""
This is the chat worker which uses the information extracted from the training data
using the EmbedTrainingProcessWorker. It finds matching answers to user queries by
computing a weighted word embedding of the user query and finding the closest match
in the training set using cosine-similarity
"""

MODEL_FILE = "model.pkl"


def _get_logger():
    logger = logging.getLogger('embedding.chat')
    return logger


class EmbeddingChatProcessWorker(ait_c.ChatProcessWorkerABC):
    def __init__(self, pool, aiohttp_client_session=None):
        super().__init__(pool)
        self.chat_args = None
        self.ai = None
        self.logger = _get_logger()
        if aiohttp_client_session is None:
            aiohttp_client_session = aiohttp.ClientSession()
        self.aiohttp_client = aiohttp_client_session
        config = SvcConfig.get_instance()
        self.w2v_client = Word2VecClient(config.w2v_server_url,
                                         self.aiohttp_client)
        self.entity_wrapper = EntityWrapper(config.er_server_url,
                                            self.aiohttp_client)
        self.cls = None

    async def start_chat(self, msg: ait_c.WakeChatMessage):
        """Handle a wake request"""
        self.logger.info("Started chat process for AI %s" % msg.ai_id)
        await self.setup_chat_session()

    async def chat_request(self, msg: ait_c.ChatRequestMessage):
        """Handle a chat request"""
        if msg.update_state:
            await self.setup_chat_session()

        # tokenize
        t_start = time.time()
        x_tokens_testset = [
            await self.entity_wrapper.tokenize(msg.question, sw_size='xlarge')
        ]
        self.logger.info("x_tokens_testset: {}".format(x_tokens_testset))
        self.logger.info("x_tokens_testset: {}".format(
            len(x_tokens_testset[0])))
        self.logger.info("tokenizing: {}s".format(time.time() - t_start))

        if x_tokens_testset[0][0] != 'UNK':
            y_pred, y_prob = await self.get_embedding_match(x_tokens_testset)
            self.logger.info("default emb: {}".format(y_pred[0]))
            self.logger.info("embedding: {}s".format(time.time() - t_start))
        else:
            y_pred = [""]
            y_prob = [0.0]

        resp = ait_c.ChatResponseMessage(msg, y_pred[0], float(y_prob[0]))
        return resp

    async def get_embedding_match(self, x_tokens_testset, msg=None, msg_spacy_entities=None):
        # get new word embeddings
        unique_tokens = list(set([w for l in x_tokens_testset for w in l]))
        unk_tokens = self.cls.get_unknown_words(unique_tokens)
        if len(unk_tokens) > 0:
            unk_words = await self.w2v_client.get_unknown_words(unk_tokens)
            self.logger.debug("unknown words: {}".format(unk_words))
            if len(unk_words) > 0:
                unk_tokens = [w for w in unk_tokens if w not in unk_words]
                x_tokens_testset = [[w for w in s if w not in unk_words]
                                    for s in x_tokens_testset]
            if len(unk_tokens) > 0:
                vecs = await self.w2v_client.get_vectors_for_words(
                    unk_tokens)
                self.cls.update_w2v(vecs)
        self.logger.debug("final tok set: {}".format(x_tokens_testset))
        # get embedding match
        y_pred, y_prob = self.cls.predict(x_tokens_testset)
        y_prob = [max(0., y_prob[0] - 0.15)]
        return y_pred, y_prob

    async def setup_chat_session(self):
        self.logger.info("Reloading model for AI %s" % self.ai_id)
        self.cls = EmbeddingComparison()
        ai_path = Path(self.ai_path)
        self.cls.load_model(ai_path / MODEL_FILE)
