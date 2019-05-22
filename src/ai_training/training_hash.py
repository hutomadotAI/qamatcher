import base64
import hashlib
import logging
from pathlib import Path


def _get_logger():
    logger = logging.getLogger('hu.training_hash')
    return logger


# read chunks of 64K from disk
MAX_READ_SIZE = 65536


def calculate_training_hash(training_file: Path) -> str:
    logger = _get_logger()
    if training_file is None or not training_file.exists():
        logger.info(
            "calculate_training_hash - training file doesn't exist: {}".format(
                str(training_file)))
        return ''

    with training_file.open(mode='rb') as file:
        md5 = hashlib.md5()
        while True:
            contents = file.read(MAX_READ_SIZE)
            if len(contents) == 0:
                break
            md5.update(contents)
        digest = md5.digest()
        digest_encoded = base64.urlsafe_b64encode(digest)
        digest_str = digest_encoded.decode("utf-8")
        return digest_str
