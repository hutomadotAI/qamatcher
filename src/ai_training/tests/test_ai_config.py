# flake8: noqa
import os
from pathlib import Path
import ai_training.ai_training_config as conf


def test_config_defaults():
    config = conf.Config()
    assert config.language == "en"
    assert config.training_data_root == "/ai"
    assert config.version is None
    assert config.chat_enabled == 1
    assert config.training_enabled == 1


def test_config_set_default_training_root_with_version():
    TEST_VERSION = "1.2.3"
    old_default_version = conf.DEFAULT_AI_VERSION
    try:
        conf.DEFAULT_AI_VERSION = TEST_VERSION
        config = conf.Config()
        assert config.version == TEST_VERSION
        assert config.training_data_root == str(Path("/ai/" + TEST_VERSION))
    finally:
        conf.DEFAULT_AI_VERSION = old_default_version


def test_set_version_env():
    TEST_VERSION = "custom_version"
    os.environ["AI_VERSION"] = TEST_VERSION
    try:
        config = conf.Config()
        config.load_from_file_and_environment("/tmp/no_file_here")
        assert config.version == TEST_VERSION
        assert config.training_data_root == str(Path("/ai/" + TEST_VERSION))
    finally:
        os.environ.clear()


def test_set_language_env():
    TEST_LANGUAGE = "es"
    os.environ["AI_LANGUAGE"] = TEST_LANGUAGE
    try:
        config = conf.Config()
        config.load_from_file_and_environment("/tmp/no_file_here")
        assert config.language == TEST_LANGUAGE
    finally:
        os.environ.clear()