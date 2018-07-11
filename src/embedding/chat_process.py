"""SVCLASSIFIER chat worker processes"""

import logging
from pathlib import Path

import ai_training.chat_process as ait_c
from spacy_wrapper import SpacyWrapper
from entity_matcher import EntityMatcher
from text_classifier_class import EmbeddingComparison
from word2vec_client import Word2VecClient
from svc_config import SvcConfig
import aiohttp

MODEL_FILE = "model.pkl"
DATA_FILE = "data.pkl"
THRESHOLD = 0.5
ENTITY_MATCH_PROBA = 0.7


def _get_logger():
    logger = logging.getLogger('embedding.chat')
    return logger


class Word2VecFailureError(Exception):
    """Failure in Word2Vec"""
    pass


class EmbeddingChatProcessWorker(ait_c.ChatProcessWorkerABC):

    def __init__(self, pool, asyncio_loop):
        super().__init__(pool, asyncio_loop)
        self.chatter = None
        self.chat_args = None
        self.ai = None
        self.logger = _get_logger()
        self.cls = None
        self.w2v_client = Word2VecClient(
            SvcConfig.get_instance().w2v_server_url)
        self.spacy_wrapper = SpacyWrapper()
        self.entity_matcher = EntityMatcher(spacy=self.spacy_wrapper)
        self.train_entities = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceb):
        if self.chatter is not None:
            self.chatter.__exit__(exc_type, exc, traceb)

    async def start_chat(self, msg: ait_c.WakeChatMessage):
        """Handle a wake request"""
        self.logger.info("Started chat process for AI %s" % msg.ai_id)
        self.setup_chat_session()

    async def chat_request(self, msg: ait_c.ChatRequestMessage):
        """Handle a chat request"""
        if msg.update_state:
            self.setup_chat_session()

        x_tokens_testset = [
            self.spacy_wrapper.tokenizeSpacy(msg.question)
        ]
        self.logger.info("x_tokens_testset: {}".format(x_tokens_testset))
        unique_tokens = list(set([w for l in x_tokens_testset for w in l]))
        unk_tokens = self.cls.get_unknown_words(unique_tokens)

        try:
            vecs = await self.w2v_client.get_vectors_for_words(unk_tokens)
        except aiohttp.client_exceptions.ClientConnectorError as exc:
            self.logger.error(
                "Could not receive response from w2v service - {}".format(exc))
            raise Word2VecFailureError()

        self.cls.update_w2v(vecs)
        yPred, yProbs = self.cls.predict(x_tokens_testset)
        if yProbs[0] < THRESHOLD:
            matched_answer = self.entity_matcher.match_entities(
                self.train_entities, msg.question)
            self.logger.info("matched_entities: {}".format(matched_answer))
            self.logger.info("train: {} test: {}".format(self.train_entities, msg.question))
            if matched_answer:
                self.logger.info("substituting {} for entity match {}".format(
                    yPred, matched_answer))
                yPred = [matched_answer]
                yProbs = [ENTITY_MATCH_PROBA]
        resp = ait_c.ChatResponseMessage(msg, yPred[0], float(yProbs[0]))
        return resp

    def setup_chat_session(self):
        self.logger.info("Reloading model for AI %s" % self.ai_id)
        self.cls = EmbeddingComparison()
        ai_path = Path(self.ai_path)
        self.cls.load_model(ai_path / MODEL_FILE)
        self.train_entities = self.entity_matcher.load_data(ai_path / DATA_FILE)
