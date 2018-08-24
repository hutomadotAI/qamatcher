"""SVCLASSIFIER chat worker processes"""

import logging
import dill
from pathlib import Path

import aiohttp

import ai_training.chat_process as ait_c
from embedding.entity_wrapper import EntityWrapper
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
            self.setup_chat_session()

        # tokenize
        x_tokens_testset = [
            await self.entity_wrapper.tokenize(msg.question, sw_size='xlarge')
        ]
        self.logger.info("x_tokens_testset: {}".format(x_tokens_testset))
        self.logger.info("x_tokens_testset: {}".format(len(x_tokens_testset[0])))

        # get question entities
        msg_entities = await self.entity_wrapper.extract_entities(msg.question)
        # self.logger.info("msg_entities: {}".format(msg_entities))

        # get string match
        sm_proba, sm_preds = await self.string_match.get_string_match(msg.question)
        if len(sm_preds) > 1:
            sm_idxs, _ = zip(*sm_preds)
            # self.logger.info("sm_idxs: {}".format(sm_idxs))
            matched_answers = self.entity_wrapper.match_entities(
                msg.question, msg_entities, subset_idxs=sm_idxs)
            if len(matched_answers) == 1:
                sm_pred = [matched_answers[0][1]]
                sm_prob = [ENTITY_MATCH_PROBA]
            elif len(matched_answers) > 1:
                sm_idxs, _ = zip(*matched_answers)
                if not any([self.string_match.train_data[i][0] == 'UNK' for i in sm_idxs]):
                    sm_pred, sm_prob = self.cls.predict(x_tokens_testset, subset_idx=sm_idxs)
                    sm_prob = [ENTITY_MATCH_PROBA+0.1]  # min(0.99, sm_prob[0])
                else:
                    sm_pred = ['']
                    sm_prob = [0.0]
            else:
                sm_pred = ['']
                sm_prob = [0.0]
        elif len(sm_preds) == 1:
            sm_idxs, sm_pred, sm_prob = [sm_preds[0][0]], [sm_preds[0][1]], [sm_proba]
        else:
            sm_pred, sm_prob = [''], [0.0]

        # entity matcher
        matched_answers = self.entity_wrapper.match_entities(
            msg.question, msg_entities)
        if len(matched_answers) == 1:
            er_pred = [matched_answers[0][1]]
            er_prob = [ENTITY_MATCH_PROBA]
        elif len(matched_answers) > 1:
            er_idxs, _ = zip(*matched_answers)
            if not any([self.string_match.train_data[i][0] == 'UNK' for i in er_idxs]):
                er_pred, er_prob = self.cls.predict(x_tokens_testset, subset_idx=er_idxs)
                er_prob = [ENTITY_MATCH_PROBA]  # min(0.99, er_prob[0])

                self.logger.info("er_pred: {} er_prob: {}".format(er_pred, er_prob))
            else:
                er_pred = ['']
                er_prob = [0.0]
        else:
            er_pred, er_prob = [''], [0.0]

        # if SM proba larger take that
        if sm_prob[0] > er_prob[0] and sm_prob[0] > STRING_PROBA_THRES:
            y_pred, y_prob = sm_pred, sm_prob
            self.logger.info("sm wins: {}".format(y_pred))
        # otherwise take ER result if there is any
        elif er_prob[0] > 0.:
            y_pred, y_prob = er_pred, er_prob
            self.logger.info("er wins: {}".format(y_pred))
        # if both ER and SM fail completely - EMB to the rescue!
        elif x_tokens_testset[0][0] != 'UNK':
            # get new word embeddings
            unique_tokens = list(set([w for l in x_tokens_testset for w in l]))
            unk_tokens = self.cls.get_unknown_words(unique_tokens)
            if len(unk_tokens) > 0:
                unk_words = await self.w2v_client.get_unknown_words(unk_tokens)
                # self.logger.info("unknown words: {}".format(unk_words))
                if len(unk_words) > 0:
                    unk_tokens = [w for w in unk_tokens if w not in unk_words]
                    x_tokens_testset = [[w for w in s if w not in unk_words] for s in x_tokens_testset]
                if len(unk_tokens) > 0:
                    vecs = await self.w2v_client.get_vectors_for_words(unk_tokens)
                    self.cls.update_w2v(vecs)
            self.logger.info("final tok set: {}".format(x_tokens_testset))
            # get embedding match
            y_pred, y_prob = self.cls.predict(x_tokens_testset)
            y_prob = [max(0., y_prob[0] - 0.15)]
            self.logger.info("default emb: {}".format(y_pred))
        else:
            y_pred = [""]
            y_prob = [0.0]

        resp = ait_c.ChatResponseMessage(msg, y_pred[0], float(y_prob[0]))
        return resp

    async def setup_chat_session(self):
        self.logger.info("Reloading model for AI %s" % self.ai_id)
        self.cls = EmbeddingComparison()
        ai_path = Path(self.ai_path)
        self.cls.load_model(ai_path / MODEL_FILE)
        self.entity_wrapper.load_data(ai_path / DATA_FILE)
        self.string_match.load_train_data(ai_path / TRAIN_FILE)
        await self.string_match.tokenize_train_data()
