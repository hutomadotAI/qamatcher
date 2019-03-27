"""SVCLASSIFIER chat worker processes"""

import logging
from pathlib import Path
import time
import aiohttp

import ai_training.chat_process as ait_c
from embedding.chat_process_base import EmbeddingChatProcessWorker
from embedding.entity_wrapper import EntityWrapperPlus
from embedding.text_classifier_class import EmbeddingComparison
from embedding.word2vec_client import Word2VecClient
from embedding.svc_config import SvcConfig
from embedding.string_match import StringMatch

MODEL_FILE = "model.pkl"
DATA_FILE = "data.pkl"
TRAIN_FILE = "train.pkl"

ENTITY_MATCH_PROBA = 0.7
STRING_PROBA_THRES = 0.45


def _get_logger():
    logger = logging.getLogger('qa_matcher.chat')
    return logger


class QAMatcherChatProcessWorker(EmbeddingChatProcessWorker):
    def __init__(self, pool, aiohttp_client_session=None):
        super().__init__(pool, aiohttp_client_session)
        self.logger = _get_logger()
        if aiohttp_client_session is None:
            aiohttp_client_session = aiohttp.ClientSession()
        self.aiohttp_client = aiohttp_client_session
        config = SvcConfig.get_instance()
        self.w2v_client = Word2VecClient(config.w2v_server_url,
                                         self.aiohttp_client)
        self.entity_wrapper = EntityWrapperPlus(config.er_server_url,
                                                self.aiohttp_client)
        self.string_match = StringMatch(self.entity_wrapper)
        self.cls = None

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

        # get question entities
        t_start = time.time()
        msg_spacy_entities = await self.entity_wrapper.extract_entities(msg.question)
        self.logger.info("msg_spacy_entities: {}".format(msg_spacy_entities))
        self.logger.info("msg_spacy_entities: {}s".format(time.time() - t_start))

        # get string match
        t_start = time.time()
        sm_pred, sm_prob = await self.get_string_match(
            x_tokens_testset, msg, msg_spacy_entities, msg.entities)
        self.logger.info("string_match: {}s".format(time.time() - t_start))

        # entity matcher
        t_start = time.time()
        er_pred, er_prob = await self.get_entity_match(x_tokens_testset, msg, msg_spacy_entities)
        self.logger.info("entity_match: {}s".format(time.time() - t_start))

        # if SM proba larger take that
        if sm_prob[0] > er_prob[0] and sm_prob[0] > STRING_PROBA_THRES:
            y_pred, y_prob = sm_pred, [min(sm_prob[0], 1.0)]
            self.logger.info("sm wins: {}".format(y_pred))
        # otherwise take ER result if there is any
        elif er_prob[0] > 0.:
            y_pred, y_prob = er_pred, er_prob
            self.logger.info("er wins: {}".format(y_pred))
        # if both ER and SM fail completely - EMB to the rescue!
        elif x_tokens_testset[0][0] != 'UNK':
            t_start = time.time()
            y_pred, y_prob = await self.get_embedding_match(
                x_tokens_testset, msg, msg_spacy_entities)
            self.logger.info("default emb: {}".format(y_pred[0]))
            self.logger.info("embedding: {}s".format(time.time() - t_start))
        else:
            y_pred = [""]
            y_prob = [0.0]

        resp = ait_c.ChatResponseMessage(msg, y_pred[0], float(y_prob[0]))
        return resp

    async def get_entity_match(self, x_tokens_testset, msg=None, msg_spacy_entities=None):
        matched_answers = self.entity_wrapper.match_entities(
            msg.question, msg_spacy_entities)
        if len(matched_answers) == 1:
            er_pred = [matched_answers[0][1]]
            er_prob = [ENTITY_MATCH_PROBA]
        elif len(matched_answers) > 1:
            er_idxs, _ = zip(*matched_answers)
            if not any(
                [self.string_match.train_data[i][0] == 'UNK'
                 for i in er_idxs]):
                er_pred, er_prob = self.cls.predict(
                    x_tokens_testset, subset_idx=er_idxs)
                er_prob = [ENTITY_MATCH_PROBA]

                self.logger.info("er_pred: {} er_prob: {}".format(
                    er_pred, er_prob))
            else:
                er_pred = ['']
                er_prob = [0.0]
        else:
            er_pred, er_prob = [''], [0.0]
        return er_pred, er_prob

    async def get_string_match(self, x_tokens_testset, msg, msg_spacy_entities, cust_ents):
        sm_proba, sm_preds = await self.string_match.get_string_match(
            msg.question, entities=cust_ents)
        if len(sm_preds) > 1:
            sm_idxs, _ = zip(*sm_preds)
            self.logger.debug("sm_idxs: {}".format(sm_idxs))
            matched_answers = self.entity_wrapper.match_entities(
                msg.question, msg_spacy_entities, subset_idxs=sm_idxs)
            if len(matched_answers) == 1:
                sm_pred = [matched_answers[0][1]]
                sm_prob = [ENTITY_MATCH_PROBA]
            elif len(matched_answers) > 1:
                sm_idxs, _ = zip(*matched_answers)
                if not any([
                        self.string_match.train_data[i][0] == 'UNK'
                        for i in sm_idxs
                ]):
                    sm_pred, sm_prob = self.cls.predict(
                        x_tokens_testset, subset_idx=sm_idxs)
                    sm_prob = [ENTITY_MATCH_PROBA + 0.1]
                    self.logger.info("sm-emb: sm_pred: {} sm_prob: {}".format(
                        sm_pred[0], sm_prob[0]
                    ))
                else:
                    sm_pred = ['']
                    sm_prob = [0.0]
            else:
                sm_pred = ['']
                sm_prob = [0.0]
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
