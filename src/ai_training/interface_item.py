"""
Abstract Base Class to be implemented by an AI Engine to represent a single
AI controlled by that Engine
(that can be controlled over HTTP)
"""

import abc
import asyncio
import copy
import datetime
import logging
import pathlib

import ai_training as ait
import ai_training.training_process as ait_t
import ai_training.chat_process as ait_c
import ai_training.common as ait_common

import async_process_pool.process_pool as a_pool

AI_TRAINING_STATUS_FILENAME = 'training_status.pkl'
DEFAULT_MAX_CHAT_LOCK_SECONDS = 10.0


class AiTrainingItemABC(abc.ABC):
    """Single AI training"""

    def __init__(self):
        self.__controller = None
        self.max_chat_lock_seconds = DEFAULT_MAX_CHAT_LOCK_SECONDS
        self.__status = ait.AiTrainingStatusWithProgress(
            ait.AiTrainingState.ai_undefined)
        self.dev_id = None
        self.ai_id = None
        self.__status_file = None
        self.last_chat_time = None
        self.training_msg = None
        self.__asyncio_condition_variable = None
        self.__training_stopped_condition_variable = None
        self.__start_training_mutex = None
        self.__shutting_down = False
        self.__update_pending = False
        self.__chat_process_pool = None
        self.__watch_training_future = None

    @property
    @abc.abstractmethod
    def logger(self) -> logging.Logger:
        """Logger"""

    @property
    @abc.abstractmethod
    def ai_data_directory(self) -> pathlib.Path:
        """The path to where this AI's data files are stored"""

    @property
    @abc.abstractmethod
    def training_pool(self):
        """Get the training pool"""

    def create_training_message(self, max_training_mins):
        """Create a training message
        - default implementation, can be overridden"""
        msg = ait_t.TrainingMessage(self.ai_data_directory, self.ai_id,
                                    max_training_mins)
        return msg

    def create_wake_chat_message(self, ai_directory, ai_id):
        """Create a wake chat message
        - default implementation, can be overridden"""
        msg = ait_c.WakeChatMessage(ai_directory, ai_id)
        return msg

    @abc.abstractmethod
    def create_chat_process_worker(self) -> (type, dict):
        """Get the chat worker - return the type to create,
           and a kwargs dictionary that will be passed
           to the chat process using set_data()"""

    # concrete methods
    @property
    def status(self) -> ait.AiTrainingStatusWithProgress:
        """Get training progress"""
        return self.__status

    @property
    def is_training_active(self) -> bool:
        """Return if is training now on this server"""
        return self.__watch_training_future is not None and not self.__watch_training_future.done()

    def update_status_from_file(self):
        """Initialize status from file"""
        ai_dir = self.ai_data_directory
        if ai_dir is None:
            self.logger.debug('No data directory set, not updating status')
            return
        self.__status_file = ai_dir / AI_TRAINING_STATUS_FILENAME

        if self.status.is_training:
            self.logger.info('update_status_from_file - is training, skipping')
            return

        self.logger.debug('update_status_from_file - loading from {}'.format(
            self.__status_file))
        file_status = ait.AiTrainingStatusWithProgress.load_safe(
            self.__status_file)
        if file_status.is_training:
            self.logger.warning(
                'update_status_from_file - file says training, '
                'overriding to queued')
            file_status.state == ait.AiTrainingState.ai_training_queued

        self.reset_status(file_status)
        self.logger.info('Updated existing AI {}/{} with status {}'.format(
            self.dev_id, self.ai_id, self.status.state))

    def initialize_status_from_file(self):
        """Initialize status from file"""
        self.__status_file = (
            self.ai_data_directory / AI_TRAINING_STATUS_FILENAME)
        self.logger.debug(
            'initialize_status_from_file - loading from {}'.format(
                self.__status_file))
        status = ait.AiTrainingStatusWithProgress.load_safe(self.__status_file)

        # reset training that is in progress last time running
        if status.state == ait.AiTrainingState.ai_training:
            new_state = ait.AiTrainingState.ai_training_queued
            self.logger.info('Resetting status from {} to {} for {}/{}'.format(
                status.state, new_state, self.dev_id, self.ai_id))
            status.state = new_state

        self.reset_status(status)
        self.logger.debug('Found Existing AI {}/{} with status {}'.format(
            self.dev_id, self.ai_id, self.status.state))

    def set_state(self, value: ait.AiTrainingState, always_save=False):
        if self.__status.state != value:
            self.__status.state = value
            self.save_status(always_save)
            self.logger.debug('Updated state of {} to {}'.format(
                self.ai_id, value))

    def reset_status(self,
                     value: ait.AiTrainingStatusWithProgress,
                     always_save=False):
        """Set training progress"""
        if self.__status != value:
            self.__status = value
            self.save_status(always_save)
            self.logger.debug('Updated status of {} to {}'.format(
                self.ai_id, value))

    def save_status(self, always_save=False):
        status = self.__status
        if status.state is ait.AiTrainingState.ai_training:
            # make sure we don't save ai_training state
            status = copy.deepcopy(self.__status)
            status.state = ait.AiTrainingState.ai_training_queued

        if self.__status_file is None:
            ai_dir = self.ai_data_directory
            if ai_dir is None:
                self.logger.debug('No data directory set, not saving status')
                return
            self.__status_file = ai_dir / AI_TRAINING_STATUS_FILENAME

        # as status is saved in shared storage, only save status if we are the
        # active training server
        if always_save:
            self.__status.save(self.__status_file)
        elif (self.controller is not None
              and self.controller.save_controller.get_save_state(self.ai_id)):
            self.__status.save(self.__status_file)
        else:
            self.logger.debug("save_status: didn't save {}/{}".format(
                self.ai_id, self.status.state))

    async def start_training(self, max_training_mins: int):
        """Start training"""
        self.logger.debug("Acquiring lock for start training of %s", self.ai_id)
        async with self.__start_training_mutex:
            self.logger.debug("Acquired lock successfully for start training of %s", self.ai_id)
            # we must clear ready to train or other state in the storage.
            # We assume that if we get a start training command we have control
            # of the shared storage area.
            self.set_state(ait.AiTrainingState.ai_training, always_save=True)

            # Track this AI's save state
            self.controller.save_controller.track_ai(self.ai_id)

            self.training_msg = self.create_training_message(max_training_mins)

            # wait for first response message
            # - only wait 1 second for queue to be available
            await self.training_pool.send_message_in(
                self.training_msg, timeout=1.0)

            # watch future messages
            coro = self._watch_training_progress()
            self.__watch_training_future = asyncio.create_task(coro)

            # notify status backend
            await self.notify_status_update()

    async def stop_training(self):
        self.logger.debug("Acquiring lock for stop training of %s", self.ai_id)
        async with self.__start_training_mutex:
            self.logger.debug("Acquired lock successfully for stop training of %s", self.ai_id)
            if not self.is_training_active:
                self.logger.info(
                    "stop_training: AI:{} not active, nothing to do".format(
                        self.ai_id))
                return

            condition = self.__training_stopped_condition_variable
            if condition is None:
                condition = asyncio.Condition()
                self.__training_stopped_condition_variable = condition
                self.logger.info(
                    "stop_training: Stopping training for AI:{}, MsgId:{}".format(
                        self.ai_id, self.training_msg.msg_id))
                await self.training_pool.send_cancel(self.training_msg)
            async with condition:
                self.logger.info(
                    "stop_training: AI:{} stop pending, waiting".format(
                        self.ai_id))
                await condition.wait()
            self.logger.info("stop_training: Training stopped for AI:{}".format(
                self.ai_id))

    def training_rejected(self):
        """Training has been rejected by API, abort"""
        # immediately stop writing to shared stoage
        self.logger.info("training_rejected: for AI:{}".format(self.ai_id))
        self.controller.save_controller.forget_ai(self.ai_id)

        # queue a new coroutine to release the status lock
        asyncio.create_task(self.stop_training())

    async def chat(self, chat_input, topic, history, ai_hash, entities):
        """Chat, take input and return a response"""

        # Check if we need to wake up
        start_chat = datetime.datetime.utcnow()
        self.logger.info("Chat in for AI %s q:%s topic:%s history:%s",
                         self.ai_id, chat_input, topic, history)

        async with self.controller.chat_lock:
            lock_time_seconds = (
                datetime.datetime.utcnow() - start_chat).total_seconds()
            if lock_time_seconds > self.max_chat_lock_seconds:
                error_message = "Chat bot overloaded: AI='{}' chat lock " \
                    "took {:.2f}, allowed {:.2f}".format(
                        self.ai_id,
                        lock_time_seconds,
                        self.max_chat_lock_seconds)
                error_data = {
                    'aiid': self.ai_id,
                    'lock_time_seconds': lock_time_seconds,
                    'max_chat_lock_seconds': self.max_chat_lock_seconds
                }
                self.logger.error(error_message, extra=error_data)
                raise ait_common.ChatOverloadedError(error_message, error_data)
            if not self.status.can_chat:
                self.update_status_from_file()
                if not self.status.can_chat:
                    raise ait_common.ChatStateError(
                        'Ai is not ready for chat', self.ai_id,
                        self.status.state)

            if ai_hash is None:
                self.logger.info(
                    "Ai hash is None for ai %s - reload training.",
                    self.ai_id)
                chat_reload_training = True
            elif ai_hash != self.status.training_hash:
                self.logger.info(
                    "Ai hash mismatch for ai %s - was '%s', "
                    "expected '%s'. Reloading...", self.ai_id, ai_hash,
                    self.status.training_hash)
                self.update_status_from_file()
                chat_reload_training = True
            else:
                chat_reload_training = False

            # check updated status is valid for chat
            # if got past the reload steps and there is a mismatch
            # in ai_hash there is a problem
            if (ai_hash is not None
                    and ai_hash != self.status.training_hash):
                raise ait_common.ChatAiHashError(
                    "Ai hash mismatch for ai {} - was '{}', expected '{}'".
                    format(self.ai_id, ai_hash, self.status.training_hash))

            result = await self.__chat_internal(chat_input, topic, history, entities,
                                                chat_reload_training)

            chat_duration_seconds = (
                datetime.datetime.utcnow() - start_chat).total_seconds()

            self.logger.info(
                "Chat out for AI %s t:%.3f(wait=%.3f)s, q:%s a:%s score:%.3f "
                "topic_out:%s history:%s",
                self.ai_id,
                chat_duration_seconds,
                lock_time_seconds,
                chat_input,
                result.answer,
                result.score,
                result.topic_out,
                result.history,
                extra={
                    'score': result.score,
                    'lock_time_seconds': lock_time_seconds,
                    'duration_seconds': chat_duration_seconds
                })
            resp = {
                'answer': result.answer,
                'score': result.score,
                'topic_out': result.topic_out,
                'history': result.history,
                'lock_time_seconds': lock_time_seconds,
                'duration_seconds': chat_duration_seconds
            }
            return resp

    async def __chat_internal(self, chat_input, topic, history, entities, chat_reload_training):
        if self.last_chat_time is None:
            await self.controller.start_chat_for_ai(self)

        if self.__chat_process_pool is None:
            process_name = 'Chat_{0}'.format(self.ai_id)
            chat_pool = a_pool.AsyncProcessPool(
                self.controller.multiprocessing_manager,
                process_name,
                num_processes=1)
            chat_process_worker, process_kwargs = \
                self.create_chat_process_worker()
            if process_kwargs is None:
                process_kwargs = {}

            await chat_pool.initialize_processes(
                chat_process_worker, **process_kwargs)
            self.__chat_process_pool = chat_pool
            msg = self.create_wake_chat_message(
                str(self.ai_data_directory), self.ai_id)
            await self.__chat_process_pool.do_work(msg)

        self.last_chat_time = datetime.datetime.utcnow()
        msg = ait_c.ChatRequestMessage(chat_input, topic, history,
                                       chat_reload_training, entities)
        result = await self.__chat_process_pool.do_work(msg)
        return result

    async def shutdown(self):
        """Chat, take input and return a response"""
        await self.stop_training()
        if not self.__shutting_down:
            self.__shutting_down = True
            await self.notify_status_update()
            await self.shutdown_chat()

    async def shutdown_chat(self):
        """Shutdown chat for this item"""
        if self.__chat_process_pool is not None:
            self.logger.info("Shutting down chat process for ai {}".format(
                self.ai_id))
            pool = self.__chat_process_pool
            self.__chat_process_pool = None
            await pool.shutdown()
        self.last_chat_time = None

    async def _watch_training_progress(self):
        training_id = self.training_msg.msg_id
        self.logger.debug("Watching messages with ID" + training_id)
        while self.status.is_training:
            try:
                msg = await self.training_pool.get_message_out_with_id(
                    training_id)
            except a_pool.JobCancelledError:
                new_state = ait.AiTrainingState.ai_error
                self.logger.info("Job cancelled for {}".format(self.ai_id))
                if self.status.state is ait.AiTrainingState.ai_training:
                    new_state = ait.AiTrainingState.ai_training_stopped
                self.set_state(new_state)
                await self.notify_status_update()
                break
            except a_pool.FailedJobError:
                self.logger.error(
                    "Job failed for {}".format(self.ai_id), exc_info=True)
                self.set_state(ait.AiTrainingState.ai_error)
                await self.notify_status_update()
                break
            except Exception:
                self.logger.error(
                    "Unexpected exception during {}".format(self.ai_id), exc_info=True)
                self.set_state(ait.AiTrainingState.ai_error)
                await self.notify_status_update()
                break
            self.status.state = msg.status
            self.status.training_progress = msg.training_completion
            self.status.training_error = msg.training_error
            self.status.training_data_hash = msg.training_data_hash
            self.save_status()
            await self.notify_status_update()

        # training is complete, save status and
        # stop being active training server
        self.controller.save_controller.forget_ai(self.ai_id)

        if self.__training_stopped_condition_variable is not None:
            # notify anyone waiting for training to stop
            async with self.__training_stopped_condition_variable:
                self.__training_stopped_condition_variable.notify_all()

        self.__training_stopped_condition_variable = None
        self.logger.info("Training completed for {}".format(self.ai_id))
        self.training_msg = None

    async def notify_status_update(self):
        if self.__asyncio_condition_variable:
            async with self.__asyncio_condition_variable:
                self.__update_pending = True
                self.logger.debug("notify_status_update for {}".format(
                    self.ai_id))
                self.__asyncio_condition_variable.notify()

    async def _send_status_update_loop(self):
        while True:
            async with self.__asyncio_condition_variable:
                if self.__update_pending:
                    self.logger.debug(
                        "_send_status_update_loop NOT waiting {}".format(
                            self.ai_id))
                else:
                    self.logger.debug(
                        "_send_status_update_loop waiting {}".format(
                            self.ai_id))
                    await self.__asyncio_condition_variable.wait()
                    self.logger.debug(
                        "_send_status_update_loop wait complete for {}".format(
                            self.ai_id))
                self.__update_pending = False

            if self.__shutting_down:
                return
            self.logger.debug(
                "_send_status_update_loop calling training_callback {}".format(
                    self.ai_id))
            await self.controller.training_callback(self)
            self.logger.debug(
                "_send_status_update_loop training_callback complete {}".
                format(self.ai_id))

    @property
    def controller(self) -> ait.AiTrainingControllerABC:
        """Controller callback object"""
        return self.__controller

    @controller.setter
    def controller(self, value: ait.AiTrainingControllerABC):
        """Controller callback object"""
        self.__controller = value

        if self.__asyncio_condition_variable is None:
            self.__asyncio_condition_variable = asyncio.Condition()
            self.__start_training_mutex = asyncio.Lock()
            coro = self._send_status_update_loop()
            asyncio.create_task(coro)
