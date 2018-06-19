"""SVCLASSIFIER chat worker processes"""

import logging
import numpy

import ai_training.chat_process as ait_c
from spacy_wrapper import SpacyWrapper
from text_classifier_class import EmbeddingComparison
from word2vec_client import Word2VecClient
from svc_config import SvcConfig
import aiohttp

MODEL_FILE = "model.pkl"


def _get_logger():
    logger = logging.getLogger('embedding.chat')
    return logger


class EmbeddingChatProcessWorker(ait_c.ChatProcessWorkerABC):

    __spacy_wrapper = None

    def __init__(self, pool, asyncio_loop):
        super().__init__(pool, asyncio_loop)
        self.chatter = None
        self.chat_args = None
        self.ai = None
        self.is_ready = False
        self.logger = _get_logger()
        self.w2v_client = Word2VecClient(
            SvcConfig.get_instance().w2v_server_url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceb):
        if self.chatter is not None:
            self.chatter.__exit__(exc_type, exc, traceb)

    async def start_chat(self, msg: ait_c.WakeChatMessage):
        """Handle a wake request"""

    async def chat_request(self, msg: ait_c.ChatRequestMessage):
        """Handle a chat request"""
        if msg.update_state or self.chatter is None:
            self.setup_chat_session()

        if self.is_ready:
            _ = msg.question.split(' ')
            question_list = []
            question_list.append(msg.question)
            x_tokens_testset = [
                EmbeddingChatProcessWorker.__spacy_wrapper.tokenizeSpacy(s)
                for s in question_list
            ]

            unique_tokens = list(set([w for l in x_tokens_testset for w in l]))
            cls = EmbeddingComparison()
            cls.load_model(self.ai_path + "/" + MODEL_FILE)
            unk_tokens = cls.get_unknown_words(unique_tokens)

            try:
                vecs = await self.w2v_client.get_vectors_for_words(unk_tokens)
            except aiohttp.client_exceptions.ClientConnectorError as exc:
                self.logger.warn(
                    "Could not receive response from w2v service - {}".format(
                        exc))
                return ait_c.ChatResponseMessage(msg, None, 0.0)

            cls.update_w2v(vecs)
            yPred, yProbs = cls.predict(x_tokens_testset)
            resp = ait_c.ChatResponseMessage(msg, yPred[0], yProbs[0].astype(str))
            return resp

        resp = ait_c.ChatResponseMessage(msg, None, 0.0)
        return resp

    def setup_chat_session(self):

        if EmbeddingChatProcessWorker.__spacy_wrapper is None:
            EmbeddingChatProcessWorker.__spacy_wrapper = SpacyWrapper()

        self.is_ready = True
