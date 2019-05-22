import logging
import configparser
import os
from pathlib import Path


def _get_logger():
    logger = logging.getLogger('hu.ai_training_config')
    return logger


DEFAULT_TRAINING_DATA_ROOT = "/ai"
DEFAULT_AI_TRAINING_ENABLED = True
DEFAULT_AI_CHAT_ENABLED = True
DEFAULT_AI_THIS_SERVER_URL = ""
DEFAULT_API_HEARTBEAT_TIMEOUT_SECONDS = 15.0
DEFAULT_API_SHUTDOWN_TIMEOUT_SECONDS = DEFAULT_API_HEARTBEAT_TIMEOUT_SECONDS * 20
DEFAULT_MAX_CHAT_LOCK_SECONDS = 10.0
DEFAULT_AI_LANGUAGE = "en"
DEFAULT_AI_VERSION = None


def calculate_training_data_root(version: str, training_data_root_base: str):
    if version:
        training_data_root = str(Path(training_data_root_base) / version)
    else:
        training_data_root = training_data_root_base
    return training_data_root


class Config:
    """Configuration file for AI training"""

    def __init__(self):
        self.logger = _get_logger()
        self.version = DEFAULT_AI_VERSION
        self.language = DEFAULT_AI_LANGUAGE
        self.training_data_root = calculate_training_data_root(
            DEFAULT_AI_VERSION, DEFAULT_TRAINING_DATA_ROOT)
        self.api_server = None
        self.training_enabled = DEFAULT_AI_TRAINING_ENABLED
        self.chat_enabled = DEFAULT_AI_CHAT_ENABLED
        self.this_server_url = DEFAULT_AI_THIS_SERVER_URL
        self.api_heartbeat_timeout_seconds = DEFAULT_API_HEARTBEAT_TIMEOUT_SECONDS
        self.api_shutdown_timeout_seconds = DEFAULT_API_SHUTDOWN_TIMEOUT_SECONDS
        self.max_chat_lock_seconds = DEFAULT_MAX_CHAT_LOCK_SECONDS

    def load_from_file_and_environment(self, config_file_path):
        self.logger.debug('Looking for config file:' + config_file_path)
        config_root = None
        if os.path.isfile(config_file_path):
            self.logger.debug('Config file found')
            config_root = configparser.ConfigParser()
            try:
                config_root.read(config_file_path)

            except (configparser.Error, KeyError):
                self.logger.error(
                    'Failed to use config file, falling back to defaults',
                    exc_info=True)
        else:
            self.logger.debug('No config file found')
        self._load_internal(config_root)

    def _load_internal(self, config_root):
        ai_config = None
        if config_root is not None:
            try:
                ai_config = config_root['AI']
            except KeyError:
                self.logger.warning('No AI section in config file')

        self.version = self.get_from_environment_or_config(
            ai_config, 'Version', 'AI_VERSION', DEFAULT_AI_VERSION)
        self.language = self.get_from_environment_or_config(
            ai_config, 'Language', 'AI_LANGUAGE', DEFAULT_AI_LANGUAGE)
        self.this_server_url = self.get_from_environment_or_config(
            ai_config, 'This server URL', 'AI_THIS_SERVER_URL',
            DEFAULT_AI_THIS_SERVER_URL)
        training_data_root_base = self.get_from_environment_or_config(
            ai_config, 'Training data root', None, DEFAULT_TRAINING_DATA_ROOT)
        self.training_data_root = calculate_training_data_root(
            self.version, training_data_root_base)

        self.api_server = self.get_from_environment_or_config(
            ai_config, 'API server', 'API_BACKEND_STATUS_ENDPOINT')

        self.training_enabled = bool(
            self.get_from_environment_or_config(
                ai_config, 'Training Enabled', 'AI_TRAINING_ENABLED',
                DEFAULT_AI_TRAINING_ENABLED, int))
        self.chat_enabled = bool(
            self.get_from_environment_or_config(ai_config, 'Chat Enabled',
                                                'AI_CHAT_ENABLED',
                                                DEFAULT_AI_CHAT_ENABLED, int))

        self.api_heartbeat_timeout_seconds = self.get_from_environment_or_config(
            ai_config, 'API Heartbeat timeout',
            'API_HEARTBEAT_TIMEOUT_SECONDS',
            DEFAULT_API_HEARTBEAT_TIMEOUT_SECONDS, float)

        # time after which the backend will restart if no heartbeat is received
        self.api_shutdown_timeout_seconds = self.get_from_environment_or_config(
            ai_config, 'API Shutdown timeout', 'API_SHUTDOWN_TIMEOUT_SECONDS',
            DEFAULT_API_SHUTDOWN_TIMEOUT_SECONDS, float)

        self.max_chat_lock_seconds = self.get_from_environment_or_config(
            ai_config, 'Max chat lock seconds', 'MAX_CHAT_LOCK_SECONDS',
            DEFAULT_MAX_CHAT_LOCK_SECONDS, float)

        self.logger.debug('AI: version %s, language %s', self.version,
                          self.language)
        self.logger.debug('AI: at %s, training_data_root %s',
                          self.this_server_url, self.training_data_root)
        self.logger.debug('AI: Reporting to API at %s', self.api_server)
        self.logger.debug('AI: Training enabled=%s, Chat enabled=%s',
                          self.training_enabled, self.chat_enabled)
        self.logger.debug('AI: Timeouts: register=%2fs, shutdown=%.1fs',
                          self.api_heartbeat_timeout_seconds,
                          self.api_shutdown_timeout_seconds)
        self.logger.debug('AI: Timeouts: register=%.2fs, shutdown=%.1fs',
                          self.api_heartbeat_timeout_seconds,
                          self.api_shutdown_timeout_seconds)

    def get_from_environment_or_config(self,
                                       config,
                                       config_key: str,
                                       env_key: str,
                                       default=None,
                                       return_type=str):
        # first read from environment key
        if env_key is not None:
            self.logger.debug("Reading environment key '{}'".format(env_key))
            env_value = os.environ.get(env_key)
            if env_value is not None:
                self.logger.debug("Env Key '{}'='{}'".format(
                    env_key, env_value))
                return return_type(env_value)

        # now read from file
        if config is not None:
            self.logger.debug("Reading config key '{}'".format(config_key))
            try:
                config_value = config[config_key]
                self.logger.debug("Config Key '{}'='{}'".format(
                    config_key, config_value))
                return return_type(config_value)
            except KeyError:
                pass
        return default
