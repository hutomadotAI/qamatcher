"""SVCLASSIFIER chat worker processes"""

import logging
from pathlib import Path

import aiohttp

import ai_training.chat_process as ait_c
from embedding.entity_wrapper import EntityWrapper
from embedding.string_match import StringMatch
from embedding.text_classifier_class import EmbeddingComparison
from embedding.word2vec_client import Word2VecClient
from embedding.svc_config import SvcConfig


MODEL_FILE = "model.pkl"
DATA_FILE = "data.pkl"
TRAIN_FILE = "train.pkl"

THRESHOLD = 0.5
ENTITY_MATCH_PROBA = 0.7
STRING_PROBA_THRES = 0.8


def _get_logger():
    logger = logging.getLogger('embedding.chat')
    return logger


class EmbeddingChatProcessWorker(ait_c.ChatProcessWorkerABC):

    def __init__(self, pool, asyncio_loop, aiohttp_client_session=None):
        super().__init__(pool, asyncio_loop)
        self.chat_args = None
        self.ai = None
        self.logger = _get_logger()
        if aiohttp_client_session is None:
            aiohttp_client_session = aiohttp.ClientSession()
        self.aiohttp_client = aiohttp_client_session
        config = SvcConfig.get_instance()
        self.w2v_client = Word2VecClient(
            config.w2v_server_url, self.aiohttp_client)
        self.entity_wrapper = EntityWrapper(config.er_server_url, self.aiohttp_client)
        self.string_match = StringMatch(self.entity_wrapper)
        self.cls = None

    async def start_chat(self, msg: ait_c.WakeChatMessage):
        """Handle a wake request"""
        self.logger.info("Started chat process for AI %s" % msg.ai_id)
        await self.setup_chat_session()

    async def chat_request(self, msg: ait_c.ChatRequestMessage):
        """Handle a chat request"""
        if msg.update_state:
            await self.setup_chat_session()

        x_tokens_testset = [
            await self.entity_wrapper.tokenize(msg.question)
        ]
        self.logger.info("x_tokens_testset: {}".format(x_tokens_testset))

        # get question entities
        msg_entities = await self.entity_wrapper.extract_entities(msg.question)
        self.logger.debug("msg_entities: {}".format(msg_entities))

        # get string match
        sm_pred, sm_prob = await self.get_string_match(msg, msg_entities, x_tokens_testset)
        if sm_prob[0] > STRING_PROBA_THRES:
            yPred = sm_pred
            yProbs = sm_prob
        else:
            unique_tokens = list(set([w for l in x_tokens_testset for w in l]))
            unk_tokens = self.cls.get_unknown_words(unique_tokens)

            vecs = await self.w2v_client.get_vectors_for_words(unk_tokens)

            self.cls.update_w2v(vecs)
            yPred, yProbs = self.cls.predict(x_tokens_testset)
            if yProbs[0] < THRESHOLD or len(x_tokens_testset) < 3:
                matched_answers = self.entity_wrapper.match_entities(
                    msg.question)
                self.logger.info("matched_entities: {}".format(matched_answers))
                if matched_answers:
                    if len(matched_answers) > 1:
                        train_idx = [e[0] for e in matched_answers]
                        yPred, yProbs = self.cls.predict(x_tokens_testset, subset_idx=train_idx)
                        self.logger.info("multiple entity matches {}; pick {}".format(
                            matched_answers, yPred))
                    else:
                        self.logger.info("substituting {} for entity match {}".format(
                            yPred, matched_answers))
                        yPred = [matched_answers[0][1]]
                        yProbs = [ENTITY_MATCH_PROBA]
        resp = ait_c.ChatResponseMessage(msg, yPred[0], float(yProbs[0]))
        return resp

    async def get_string_match(self, msg, msg_entities, x_tokens_testset):
        # get string match
        sm_proba, sm_preds = await self.string_match.get_string_match(
            msg.question)
        if len(sm_preds) > 1:
            sm_idxs, _ = zip(*sm_preds)
            sm_pred, sm_prob = self.cls.predict(
                x_tokens_testset, subset_idx=sm_idxs)
            sm_prob = [sm_proba]
        elif len(sm_preds) == 1:
            sm_idxs, sm_pred, sm_prob = [sm_preds[0][0]], [sm_preds[0][1]], [
                sm_proba
            ]
        else:
            sm_pred, sm_prob = [''], [0.0]

        return sm_pred, sm_prob

    async def setup_chat_session(self):
        self.logger.info("Reloading model for AI %s" % self.ai_id)
        self.cls = EmbeddingComparison()
        ai_path = Path(self.ai_path)
        self.cls.load_model(ai_path / MODEL_FILE)
        self.entity_wrapper.load_data(ai_path / DATA_FILE)
        self.string_match.load_train_data(ai_path / TRAIN_FILE)
        await self.string_match.tokenize_train_data()
