"""SV Classifier server code"""
import asyncio
import logging
import logging.config
import os
import pathlib

from aiohttp import web
import yaml

from emb_train.training_process import EmbedTrainingProcessWorker
from emb_common.svc_config import SvcConfig
from hu_http_train.interface_item import TrainItemABC
from hu_http_train.interface import AiTrainingProviderABC
from hu_http_common.ai_training_config import Config
import hu_http_train.http_server as http_server


def _get_logger():
    logger = logging.getLogger('embedding.server')
    return logger


class EmbeddingAiItem(TrainItemABC):
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


class EmbeddingAiProvider(AiTrainingProviderABC):
    """Similarity class"""

    def __init__(self, config):
        super().__init__()
        self.__config = config
        self.training_pool = None
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

        # For training servers only, create storage directory if doesn't exist
        training_root = pathlib.Path(self.config.training_data_root)
        if not training_root.exists():
            self.logger.warning("Directory %s doesn't exist, creating...",
                                training_root)
            training_root.mkdir(parents=True, exist_ok=True)

        training_processes = 1
        training_queue_size = training_processes * 2
        self.training_pool = await self.controller.create_training_process_pool(
            training_processes, training_queue_size,
            EmbedTrainingProcessWorker)

        asyncio.create_task(self.__log_loop_tasks())

    async def on_shutdown(self):
        """Shutdown SVCLASS worker processes"""
        await super().on_shutdown()
        if self.process_pool2 is not None:
            await self.process_pool2.shutdown()
        if self.training_pool is not None:
            await self.training_pool.shutdown()

    def training_item_factory(self, dev_id, ai_id) -> TrainItemABC:
        """Called when need to create a new training item"""
        item = EmbeddingAiItem(self, dev_id, ai_id)
        return item

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
    http_server.initialize_ai_training_http(app, ai_provider)


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
        (): emb_train.server.EmbLogFilter
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
        logging_config = yaml.load(LOGGING_CONFIG_TEXT)
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
