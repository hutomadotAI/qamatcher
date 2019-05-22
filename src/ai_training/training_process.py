import abc
import asyncio
import datetime
import logging
from pathlib import Path

import ai_training as ait
import ai_training.save_controller as save_controller

import async_process_pool.process_pool as a_pool

AI_TRAINING_STATUS_FILENAME = 'training_status.pkl'


def _get_logger():
    logger = logging.getLogger('hu.ai_training.training_process')
    return logger


class TrainingMessage(a_pool.Message):
    """Message class for triggering training an Ai.
    Subclass this message type if you need to pass more information in it"""

    def __init__(self, ai_path, ai_id, max_training_mins: int):
        super().__init__()
        self.ai_path = ai_path
        self.ai_id = ai_id
        self.max_training_mins = max_training_mins


class StatusResponse(a_pool.Response):
    """Message response class for reporting the training status of an AI"""

    def __init__(self,
                 msg_in_response_to: a_pool.Message,
                 ai_id,
                 status,
                 training_completion=None,
                 training_error=None,
                 training_data_hash=None):
        super().__init__(msg_in_response_to)
        self.ai_id = ai_id
        self.status = status
        self.training_completion = training_completion
        self.training_error = training_error
        self.training_data_hash = training_data_hash


SAVE_WAIT_TIME_SECONDS = 1.0


class StatusCallback:
    """Callback used inside training code to report progress and check for cancellation"""

    def __init__(self, training_message, pool,
                 save_controller: save_controller.SaveController, ai_id):
        self.training_message = training_message
        self.pool = pool
        self.save_controller = save_controller
        self.ai_id = ai_id
        self.last_progress = StatusResponse(self.training_message, self.ai_id,
                                            ait.AiTrainingState.ai_training,
                                            0.0, None)

    def report_progress(self,
                        training_progress,
                        training_error=None,
                        training_data_hash=None):
        resp = StatusResponse(
            self.training_message, self.ai_id, ait.AiTrainingState.ai_training,
            training_progress, training_error, training_data_hash)
        self.last_progress = resp
        self.pool.send_message_out_sync(resp)

    def check_for_cancel(self):
        self.pool.check_for_cancel(self.training_message)

    def can_save(self):
        self.check_for_cancel()
        value = self.save_controller.get_save_state(self.ai_id)
        if not value:
            # make sure we resend an update message to API to ensure we don't wait
            # forever if API is temporarily offline.
            # Messages are throttled at a higher level,
            # so we don't need to worry about overloading API
            self.pool.send_message_out_sync(self.last_progress)
        return value

    async def wait_to_save(self):
        logger = _get_logger()
        logged = False
        while not self.can_save():
            if not logged:
                logger.info(
                    "wait_to_save: not able to save for {}, waiting...".format(
                        self.ai_id))
                logged = True

            await asyncio.sleep(SAVE_WAIT_TIME_SECONDS)

        if logged:
            logger.info("wait_to_save: now able to save for {}".format(
                self.ai_id))


class TrainingProcessWorkerABC(a_pool.ProcessWorkerABC):
    def __init__(self, pool):
        super().__init__(pool)
        self.callback_object = None
        self.logger = _get_logger()
        self.save_controller = None

    async def process_message(self, msg):
        if isinstance(msg, TrainingMessage):
            await self.process_training_message(msg)

    @a_pool.job_runner
    async def process_training_message(self, msg: TrainingMessage):
        """Train the model"""
        self.logger.info("Training started for {}".format(msg.ai_id))
        self.pool.check_for_cancel(msg)
        resp = StatusResponse(msg, msg.ai_id, ait.AiTrainingState.ai_training,
                              0.0, None)
        await self.pool.send_message_out(resp)

        self.callback_object = StatusCallback(
            msg, self.pool, self.save_controller, msg.ai_id)

        train_start = datetime.datetime.utcnow()

        try:
            # check for blank training
            training_file = Path(
                msg.ai_path) / ait.AI_TRAINING_STANDARD_FILE_NAME
            topic = ait.file_load_training_data_v1(training_file)
            if topic.is_empty():
                self.logger.warning(
                    "Training data empty for AI: {}, marking as complete".
                    format(msg.ai_id))
                final_status = ait.AiTrainingState.ai_training_complete
                training_data_hash = None
                empty_state = ait.AiTrainingStatusWithProgress(final_status)
                # wait to have permission to write the save status
                await self.callback_object.wait_to_save()
                empty_state.save(
                    Path(msg.ai_path) / AI_TRAINING_STATUS_FILENAME)
            else:
                final_status, training_data_hash = await self.train(
                    msg, topic, self.callback_object)
        finally:
            self.callback_object = None

        train_duration = datetime.datetime.utcnow() - train_start
        self.logger.info(
            "Training ended for {}, status:{}, duration:{}, hash:{}".format(
                msg.ai_id, final_status, train_duration, training_data_hash),
            extra={
                "ai_id": msg.ai_id,
                "status": str(final_status),
                "duration": str(train_duration),
                "hash": training_data_hash
            })

        resp = StatusResponse(
            msg,
            msg.ai_id,
            final_status,
            training_data_hash=training_data_hash)
        return resp

    @abc.abstractmethod
    async def train(self, msg, topic: ait.Topic, callback_object
                    ) -> (ait.AiTrainingStatusWithProgress, str):
        """Training function"""

    def set_data(self, data):
        """Set data - any extra multiprocess data that was passed at start"""
        self.save_controller = data['save_controller']
