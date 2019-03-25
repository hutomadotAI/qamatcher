"""SV Classifier server code"""
import asyncio
import logging
import logging.config
import os
import pathlib

from aiohttp import web
import yaml
import async_process_pool

from hu_http_chat.interface_item import ChatItemABC
from hu_http_chat.interface import AiTrainingProviderABC
from hu_http_common.ai_training_config import Config
from hu_http_chat.http_server import initialize_ai_training_http

from emb_chat.chat_process import EmbeddingChatProcessWorker
from emb_common.svc_config import SvcConfig


def _get_logger():
    logger = logging.getLogger('embedding.server')
    return logger


class EmbeddingAiItem(ChatItemABC):
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

    def create_chat_process_worker(self) -> (type, dict):
        """Get the chat worker - return the type to create"""
        return EmbeddingChatProcessWorker, {
            'process_pool': self.ai_provider.process_pool2
        }


class EmbeddingAiProvider(AiTrainingProviderABC):
    """Similarity class"""

    def __init__(self, config):
        super().__init__()
        self.__config = config
        self.process_pool2 = None
        self.logger = _get_logger()

    # other required methods in the ABC
    @property
    def ai_engine_name(self):
        return "emb"

    @property
    def config(self):
        return self.__config

    async def on_startup(self):
        """Initialize SVCLASS worker processes"""

        chat_processes = 1 if self.config.chat_enabled else 0
        if chat_processes > 0:
            calc_queue_size = chat_processes * 2
            self.process_pool2 = async_process_pool.process_pool.AsyncProcessPool(
                self.controller.multiprocessing_manager, 'EMBEDDING_Calc',
                chat_processes, calc_queue_size, calc_queue_size)
            await self.process_pool2.initialize_processes(
                EmbeddingChatProcessWorker)

        asyncio.create_task(self.__log_loop_tasks())

    async def on_shutdown(self):
        """Shutdown SVCLASS worker processes"""
        await super().on_shutdown()
        if self.process_pool2 is not None:
            await self.process_pool2.shutdown()

    async def __log_loop_tasks(self):
        while True:
            pending_tasks = len(asyncio.all_tasks())
            self.logger.info(
                "asyncio tasks pending count = %d",
                pending_tasks,
                extra={"tasks": pending_tasks})
            await asyncio.sleep(5)


def load_svm_config_from_environment():
    """Load SVM configuration frm file/environment"""
    config = Config()
    config.load_from_file_and_environment("emb.config")
    return config


def init_aiohttp(app, config=None):
    """Initialize aiohttp"""
    ai_provider = EmbeddingAiProvider(config)
    initialize_ai_training_http(app, ai_provider)


LOGGING_CONFIG_TEXT = """
version: 1
root:
  level: DEBUG
  handlers: ['console']
formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "(asctime) (levelname) (name) (message)"
filters:
    emblogfilter:
        (): emb_chat.server.EmbLogFilter
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    stream: ext://sys.stdout
    formatter: json
    filters: [emblogfilter]
"""


class EmbLogFilter(logging.Filter):
    def __init__(self):
        self.language = os.environ.get("AI_LANGUAGE", "en")
        self.version = os.environ.get("AI_VERSION", None)

    def filter(self, record):
        """Add language, and if available, the version"""
        record.emb_language = self.language
        if self.version:
            record.emb_version = self.version
        return True


def main():
    """Main function"""
    logging_config_file = os.environ.get("LOGGING_CONFIG_FILE", None)
    if logging_config_file:
        logging_config_path = pathlib.Path(logging_config_file)
        with logging_config_path.open() as file_handle:
            logging_config = yaml.safe_load(file_handle)
    else:
        logging_config = yaml.safe_load(LOGGING_CONFIG_TEXT)
    print("*** LOGGING CONFIG ***")
    print(logging_config)
    print("*** LOGGING CONFIG ***")
    logging.config.dictConfig(logging_config)

    app = web.Application()
    init_aiohttp(app, load_svm_config_from_environment())

    logger = _get_logger()
    port = SvcConfig.get_instance().server_port
    logger.info("Starting embedding server", extra={"port": port})
    web.run_app(app, port=port)


if __name__ == '__main__':
    main()
