"""Common chat worker processes logic"""
import abc
from pathlib import Path
import logging

import ai_training as ait

import async_process_pool.process_pool as a_pool


def _get_logger():
    logger = logging.getLogger('hu.ai_training.chat_process')
    return logger


class WakeChatMessage(a_pool.Message):
    """Message class for wake request of a chat process"""

    def __init__(self, ai_path: Path, ai_id: str):
        super().__init__()
        self.ai_path = ai_path
        self.ai_id = ai_id


class WakeChatResponse(a_pool.Response):
    """Response class for wake request of a chat process"""

    def __init__(self, msg_in_response_to: WakeChatMessage):
        super().__init__(msg_in_response_to)


class ChatRequestMessage(a_pool.Message):
    """Message class for chat"""

    def __init__(self, question, topic_in, history, update_state, entities):
        super().__init__()
        self.question = question
        self.topic_in = topic_in
        self.history = history
        self.update_state = update_state
        self.entities = entities


class ChatResponseMessage(a_pool.Response):
    """Response class for chat"""

    def __init__(self,
                 msg_in_response_to: ChatRequestMessage,
                 answer,
                 score,
                 topic_out=None,
                 history=None):
        super().__init__(msg_in_response_to)
        self.answer = answer
        self.score = score
        self.topic_out = topic_out
        self.history = history


class ChatProcessWorkerABC(a_pool.ProcessWorkerABC):
    def __init__(self, pool):
        super().__init__(pool)
        self.ai_id = None
        self.ai_path = None
        self.training_blank = False
        self.logger = _get_logger()

    async def process_message(self, msg):
        if isinstance(msg, WakeChatMessage):
            await self.process_start_chat(msg)
        elif isinstance(msg, ChatRequestMessage):
            await self.process_chat_request(msg)

    @a_pool.job_runner
    async def process_start_chat(self, msg: WakeChatMessage):
        """Handle a wake request"""
        self.ai_id = msg.ai_id
        self.ai_path = msg.ai_path

        # check for blank training
        topic = None
        training_file = Path(msg.ai_path) / ait.AI_TRAINING_STANDARD_FILE_NAME
        if training_file.exists():
            topic = ait.file_load_training_data_v1(training_file)
        if topic is not None and topic.is_empty():
            self.logger.warning(
                "Training data empty for AI: {}, marking as blank chat".format(
                    msg.ai_id))
            self.training_blank = True
        else:
            await self.start_chat(msg)

        resp = WakeChatResponse(msg)
        return resp

    @a_pool.job_runner
    async def process_chat_request(self, msg: ChatRequestMessage):
        """Handle a chat request"""
        if self.training_blank:
            self.logger.info(
                "Training data empty for AI: {}, returning blank response".
                format(self.ai_id))
            resp = ChatResponseMessage(msg, None, 0.0)
        else:
            try:
                resp = await self.chat_request(msg)
            except ait.TrainingNotFoundError:
                logger = _get_logger()
                logger.warning("Training not found for {}".format(self.ai_id))
                resp = ChatResponseMessage(msg, None, 0.0)
        return resp

    @abc.abstractmethod
    async def start_chat(self, msg: WakeChatMessage):
        """Start chat"""

    @abc.abstractmethod
    async def chat_request(self,
                           msg: ChatRequestMessage) -> ChatResponseMessage:
        """Chat request"""
