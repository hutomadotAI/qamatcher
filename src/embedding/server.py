"""SV Classifier server code"""
import asyncio
import logging
import logging.config
import pathlib
import os

import asyncio_utils
from aiohttp import web
import yaml

import ai_training as ait
from chat_process import EmbeddingChatProcessWorker
from training_process import EmbedTrainingProcessWorker
from svc_config import SvcConfig


def _get_logger():
    logger = logging.getLogger('embedding.server')
    return logger


class EmbeddingAiItem(ait.AiTrainingItemABC):
    def __init__(self, wnet_ai_provider, dev_id, ai_id):
        super().__init__()
        self.dev_id = dev_id
        self.ai_id = ai_id
        self.ai_provider = wnet_ai_provider
        self.__logger = _get_logger()
        data_root = pathlib.Path(wnet_ai_provider.config.training_data_root)
        self.__ai_data_directory = data_root / dev_id / ai_id
        self.training_msg = None
        self.initialize_status_from_file()

    @property
    def logger(self):
        return self.__logger

    @property
    def ai_data_directory(self) -> pathlib.Path:
        return self.__ai_data_directory

    @property
    def training_pool(self):
        """Get the training pool"""
        return self.ai_provider.training_pool

    def create_chat_process_worker(self) -> (type, dict):
        """Get the chat worker - return the type to create"""
        return EmbeddingChatProcessWorker, {
            'process_pool': self.ai_provider.process_pool2
        }


class EmbedingAiProvider(ait.AiTrainingProviderABC):
    """Similarity class"""

    def __init__(self, config):
        super().__init__()
        self.__config = config
        self.process_pool2 = None
        self.training_pool = None

    # other required methods in the ABC
    @property
    def ai_engine_name(self):
        return "emb"

    @property
    def config(self):
        return self.__config

    async def on_startup(self):
        """Initialize SVCLASS worker processes"""
        asyncio_loop = self.controller.asyncio_loop
        thread_pool_executor = self.controller.thread_pool_executor
        ai_list = await asyncio_loop.run_in_executor(
            thread_pool_executor, ait.find_training_from_directory,
            self.config.training_data_root)

        for (dev_id, ai_id) in ai_list:
            self.create(dev_id, ai_id)

        training_processes = 1 if self.config.training_enabled else 0
        if training_processes > 0:
            training_queue_size = training_processes * 2
            self.training_pool = await self.controller.create_training_process_pool(
                training_processes, training_queue_size,
                EmbedTrainingProcessWorker)

        chat_processes = 1 if self.config.chat_enabled else 0
        if chat_processes > 0:
            calc_queue_size = chat_processes * 2
            self.process_pool2 = asyncio_utils.AsyncProcessPool(
                self.controller.multiprocessing_manager, 'EMBEDDING_Calc',
                asyncio_loop, chat_processes, calc_queue_size, calc_queue_size)
            await self.process_pool2.initialize_processes(
                EmbeddingChatProcessWorker)

    async def on_shutdown(self):
        """Shutdown SVCLASS worker processes"""
        await super().on_shutdown()
        if self.process_pool2 is not None:
            await self.process_pool2.shutdown()
        if self.training_pool is not None:
            await self.training_pool.shutdown()

    def training_item_factory(self, dev_id, ai_id) -> ait.AiTrainingItemABC:
        """Called when need to create a new training item"""
        item = EmbeddingAiItem(self, dev_id, ai_id)
        return item


def load_svm_config_from_environment():
    """Load SVM configuration frm file/environment"""
    config = ait.Config()
    config.load_from_file_and_environment("emb.config")
    return config


def init_aiohttp(app, loop, config=None):
    """Initialize aiohttp"""
    ai_provider = EmbedingAiProvider(config)
    ait.initialize_ai_training_http(app, ai_provider, loop)


LOGGING_CONFIG_TEXT = """
version: 1
root:
  level: DEBUG
  handlers: ['console' ,'elastic']
formatters:
  default:
    format: "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s"
    datefmt: "%Y%m%d_%H%M%S"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    stream: ext://sys.stdout
    formatter: default
  elastic:
    class: hu_logging.HuLogHandler
    level: INFO
    log_path: /tmp/hu_log
    log_tag: EMB
    es_log_index: ai-embedding-v1
    multi_process: False
"""


def main():
    """Main function"""
    logging_config = yaml.load(LOGGING_CONFIG_TEXT)
    logging_config['handlers']['elastic']['elastic_search_url'] = \
        os.environ.get('LOGGING_ES_URL', None)
    log_tag = os.environ.get('LOGGING_ES_TAG', None)
    if log_tag:
        logging_config['handlers']['elastic']['log_tag'] = log_tag
    logging.config.dictConfig(logging_config)

    loop = asyncio.get_event_loop()
    app = web.Application()
    init_aiohttp(app, loop, load_svm_config_from_environment())

    web.run_app(app, port=SvcConfig.get_instance().server_port)


if __name__ == '__main__':
    main()
