"""
Utilities for handling training files
"""
import logging
from pathlib import Path
import uuid

import ai_training

from ai_training.training_hash import calculate_training_hash

AI_TRAINING_STANDARD_FILE_NAME = "training_combined.txt"


def _get_logger():
    logger = logging.getLogger('hu.training_file')
    return logger


def check_training_exists(training_root_directory, dev_id, ai_id) -> bool:
    """Check if training exists for a given Dev/AI"""
    root_path = Path(training_root_directory)
    training_file = root_path / dev_id / ai_id / AI_TRAINING_STANDARD_FILE_NAME
    if training_file.exists():
        return True
    return False


def find_training_from_directory(training_root_directory):
    """Walk file system from root training directory to find valid AI training
    data"""
    logger = _get_logger()
    logger.info('Scanning for AI training files in ' + training_root_directory)
    root_path = Path(training_root_directory)
    ai_list = []
    dev_id_dirs = (sub for sub in root_path.iterdir() if sub.is_dir())
    for dev_id_dir in dev_id_dirs:
        # reject all dev directories that are not UUIDs
        try:
            uuid.UUID(dev_id_dir.name)
        except ValueError:
            logger.warning(
                'Skipping invalid DEV_ID directory: {}'.format(dev_id_dir))
            continue

        dev_id = dev_id_dir.stem
        ai_id_dirs = (sub for sub in dev_id_dir.iterdir() if sub.is_dir())
        for ai_id_dir in ai_id_dirs:
            # reject all AI directories that are not UUIDs
            try:
                uuid.UUID(ai_id_dir.name)
            except ValueError:
                logger.warning(
                    'Skipping invalid AI_ID directory: {}'.format(ai_id_dir))
                continue

            ai_id = ai_id_dir.stem
            training_file = ai_id_dir / AI_TRAINING_STANDARD_FILE_NAME
            if training_file.exists():
                ai_list.append((dev_id, ai_id))
    return ai_list


def delete_directory_and_children(path_to_clean: Path):
    """Delete a directory"""
    is_empty = True
    logger = _get_logger()
    if path_to_clean.exists():
        # walk the directory tree
        for item in path_to_clean.iterdir():
            if item.is_dir():
                # if directory, recursively enter
                is_sub_dir_empty = delete_directory_and_children(item)
                is_empty = is_empty and is_sub_dir_empty
            else:
                # if file, delete it
                try:
                    item.unlink()
                except OSError:
                    # detect previous .nfs files and ignore them
                    if item.name.startswith('.nfs'):
                        logger.warning(
                            "Failed to delete file '{}'".format(item))
                        is_empty = False
                    else:
                        raise

        # when here directory will be empty, so can call rmdir()
        if is_empty:
            try:
                path_to_clean.rmdir()
            except OSError:
                # If file is in use over NFS then a .nfs... file will be
                # present and the rmdir will fail
                logger.warning(
                    "Failed to delete directory '{}'".format(path_to_clean))
                is_empty = False

    return is_empty


def delete_ai_files(training_root_directory, dev_id, ai_id):
    """Delete AI training files"""
    root_path = Path(training_root_directory)
    ai_path = root_path / dev_id / ai_id
    logger = _get_logger()
    logger.warning('Deleting AI files at {}'.format(ai_path))
    delete_directory_and_children(ai_path)


def write_training_data_to_disk_v1(ai_path: Path, training_data: str):
    """Write training file V1 to disk as q_ and a_ files
       Separate topic files are not supported by this"""
    logger = _get_logger()
    logger.info("Training data loaded of size {}".format(len(training_data)))

    # validate the training data
    ai_training.str_load_training_data_v1(training_data)

    # Create training directory if doesn't exist
    # DON'T delete the existing training files as that will kill any new chat requests
    combined_path = ai_path / AI_TRAINING_STANDARD_FILE_NAME
    if not ai_path.exists():
        ai_path.mkdir(parents=True)

    with combined_path.open('w', encoding='utf-8') as combined_fp:
        combined_fp.write(training_data)

    hash_value = calculate_training_hash(combined_path)
    return hash_value
