"""Common files for ai training testing"""
# flake8: noqa

import asyncio
import logging
import time
import pathlib
import os
import tempfile
import json

import ai_training as ait
import ai_training.training_process as ait_t
import ai_training.chat_process as ait_c

import pytest

import logging.config
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple_formatter': {
            'format':
            "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s",
            'datefmt':
            "%Y%m%d_%H%M%S"
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple_formatter',
            'level': 'DEBUG'
        },
    },
    'loggers': {},
    'root': {
        'handlers': ['console'],
        'level': 'DEBUG',
    },
}
logging.config.dictConfig(LOGGING)


def _get_logger():
    logger = logging.getLogger('hu.ai_training.test')
    return logger


class ConfTestException(Exception):
    pass


class MockTrainingProcessWorker(ait_t.TrainingProcessWorkerABC):
    def __init__(self, pool):
        super().__init__(pool)

    def write_data(self, file_to_write):
        with file_to_write.open(mode='w') as fp:
            fp.write("File write")

    async def train(self, msg, topic, callback: ait_t.StatusCallback):
        """Mock training function"""
        ai_path = pathlib.Path(msg.ai_path)
        file1 = ai_path / '1.txt'
        file2 = ai_path / '2.txt'
        training_progress = 0.0

        while training_progress < 1.0:
            callback.check_for_cancel()
            time.sleep(0.05)
            if callback.can_save():
                self.write_data(file1)
            training_progress += 0.1
            if training_progress > 1.0:
                training_progress = 1.0
            else:
                callback.report_progress(
                    training_progress,
                    training_error=0.1,
                    training_data_hash="123")

        await callback.wait_to_save()
        self.write_data(file2)
        return (ait.AiTrainingState.ai_training_complete, "hash_goes_here")


class MockChatProcessWorker(ait_c.ChatProcessWorkerABC):
    def __init__(self, pool):
        super().__init__(pool)
        self.history = None
        self.raise_training_not_found = False

    async def start_chat(self, msg: ait_c.WakeChatMessage):
        """Handle a wake request"""
        pass

    async def chat_request(self, msg: ait_c.ChatRequestMessage):
        """Handle a chat request"""
        if self.raise_training_not_found:
            raise ait.TrainingNotFoundError
        answer = "really, " + msg.question
        score = 0.5
        # make chat take a small amount of time so we can test simultaneous chats
        await asyncio.sleep(0.1)

        # Minor cheat - change the output if entity data is present
        if msg.entities is not None:
            topic_out = json.dumps(msg.entities)
        else:
            topic_out = "The British Weather"
        resp = ait_c.ChatResponseMessage(msg, answer, score, topic_out,
                                         self.history)
        return resp

    def set_data(self, data_dict):
        self.history = data_dict['history']
        self.raise_training_not_found = data_dict['raise_training_not_found']


class MockTrainingItem(ait.AiTrainingItemABC):
    def __init__(self,
                 mock_training_provider,
                 training_data_root,
                 dev_id,
                 ai_id,
                 status=ait.AiTrainingStatusWithProgress(
                     ait.AiTrainingState.ai_undefined)):
        super().__init__()
        self.dev_id = dev_id
        self.ai_id = ai_id
        self.mock_training_provider = mock_training_provider
        self.__ai_data_directory = pathlib.Path(
            str(training_data_root)) / dev_id / ai_id
        self.__ai_data_directory.mkdir(exist_ok=True, parents=True)
        self.__training_data = None
        self.exception_on_access = False
        self.reset_status(status)

    @property
    def logger(self):
        return _get_logger()

    @property
    def ai_data_directory(self) -> pathlib.Path:
        """The path to where this AI's data files are stored"""
        if self.exception_on_access:
            raise ConfTestException("Raising an exception on request")
        return self.__ai_data_directory

    @property
    def training_pool(self):
        """Get the training pool"""
        return self.mock_training_provider.training_pool

    def create_chat_process_worker(self):
        return MockChatProcessWorker, {
            'history':
            self.mock_training_provider.chat_history,
            'raise_training_not_found':
            self.mock_training_provider.raise_training_not_found
        }


class MockTraining(ait.AiTrainingProviderABC):
    def __init__(self):
        super().__init__()
        self.training_root_dir = tempfile.TemporaryDirectory()
        self.__api_url = None
        self.__config = ait.Config()
        self.__config.chat_enabled = True
        # make training capacity 3 as there is already one AI in training
        self.__config.training_enabled = True
        self.__config.api_heartbeat_timeout_seconds = 3
        self.__config.api_shutdown_timeout_seconds = 4
        self.__config.max_chat_lock_seconds = 2.0
        self.training_pool = None
        self.chat_history = None
        self.raise_training_not_found = False
        self.is_killed = False
        self.is_killed_event = None

    # other required methods in the ABC
    @property
    def ai_engine_name(self):
        return "MOCK"

    @property
    def config(self):
        return self.__config

    async def on_startup(self):
        training_processes = 1 if self.config.training_enabled else 0 
        training_queue_size = training_processes * 2
        self.training_pool = await self.controller.create_training_process_pool(
            training_processes, training_queue_size, MockTrainingProcessWorker)

    async def on_shutdown(self):
        await super().on_shutdown()
        await self.training_pool.shutdown()

    def kill_running_process(self):
        """Test override of kill function. Tests can wait to see if the event fired
        as triggered by the shutdown watchdog"""
        if self.is_killed_event:
            self.is_killed_event.set()
        self.is_killed = True

    async def load_training_data_async(self, dev_id, ai_id) -> ait.AiTrainingItemABC:
        """Load training data - overridden to use in-memory values for the mock"""
        return self.lookup_item(dev_id, ai_id)

    def training_item_factory(self, dev_id, ai_id) -> ait.AiTrainingItemABC:
        item = MockTrainingItem(self, self.training_root_dir.name, dev_id,
                                ai_id)
        return item

    def add_training_with_state(self,
                                dev_id,
                                ai_id,
                                state,
                                exception_on_access=False):
        item = self.create(dev_id, ai_id)
        item.set_state(state)
        item.exception_on_access = exception_on_access
        item.status.training_file_hash = "file_hash"
        item.status.training_data_hash = "data_hash"

    async def set_api_server(self, api_server):
        api_server_str = str(api_server)
        self.config.api_server = api_server_str.strip('/')
        self.controller.start_registration_with_api()

    def set_chat_enabled(self, chat_enabled: bool):
        self.config.chat_enabled = chat_enabled


@pytest.fixture
def mock_training():
    mock_training = MockTraining()
    # add some dummy AIs
    mock_training.add_training_with_state(
        "d1a", "a1a", ait.AiTrainingState.ai_training_queued)
    mock_training.add_training_with_state(
        "d1", "a1", ait.AiTrainingState.ai_training)
    mock_training.add_training_with_state(
        "d2", "a2", ait.AiTrainingState.ai_ready_to_train)
    mock_training.add_training_with_state(
        "d3", "a3", ait.AiTrainingState.ai_undefined)
    mock_training.add_training_with_state(
        "d4", "a4", ait.AiTrainingState.ai_training_stopped)
    mock_training.add_training_with_state(
        "d5", "a5", ait.AiTrainingState.ai_training_complete)
    mock_training.add_training_with_state("d6", "a6",
                                          ait.AiTrainingState.ai_error)
    for ii in range(20):
        mock_training.add_training_with_state(
            "d_ready", "ai_{}".format(ii),
            ait.AiTrainingState.ai_training_complete)
    return mock_training

TRAINING_FILE_PATH = str(
    os.path.dirname(os.path.realpath(__file__)) + '/data/training_combined.txt'
)
