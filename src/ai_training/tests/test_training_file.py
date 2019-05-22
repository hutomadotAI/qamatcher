"""Tests for ai training"""
# pylint: skip-file
# flake8: noqa

import os
import ai_training.training_file as tf
import ai_training.common as common
from pathlib import Path
import tempfile
import uuid

import pytest

TRAINING_ROOT_DIR = str(os.path.dirname(os.path.realpath(__file__)) + '/data')


def test_scan_training():
    ai_list = tf.find_training_from_directory(TRAINING_ROOT_DIR)

    # The training file in lost+found will be ignored as it's not a correct name
    assert len(ai_list) == 1
    for (dev_id, ai_id) in ai_list:
        print('Found {} {}'.format(dev_id, ai_id))


def test_parse_1():
    data = """
hi
hello
"""
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        tf.write_training_data_to_disk_v1(root / '123' / '456', data)

        training_file = root / '123' / '456' / 'training_combined.txt'
        assert training_file.exists()


def test_parse_2():
    data = """
hi
hello

howdy
partner
"""
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        tf.write_training_data_to_disk_v1(root / '123' / '456', data)

        training_file = root / '123' / '456' / 'training_combined.txt'
        assert training_file.exists()


def test_parse_blank():
    data = """"""
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        tf.write_training_data_to_disk_v1(root / '123' / '456', data)

        training_file = root / '123' / '456' / 'training_combined.txt'
        assert training_file.exists()


def test_parse_bad_syntax_1():
    data = """
hi
"""
    with pytest.raises(common.TrainingSyntaxError):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tf.write_training_data_to_disk_v1(root / '123' / '456', data)


def test_parse_topic_syntax_1():
    data = """
@topic_out=blah
hi
hello

hello
hi
"""
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        tf.write_training_data_to_disk_v1(root / '123' / '456', data)

        training_file = root / '123' / '456' / 'training_combined.txt'
        assert training_file.exists()


def test_parse_topic_syntax_at_file_top():
    data = """@topic_out=blah
hi
hello

hello
hi
"""
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        tf.write_training_data_to_disk_v1(root / '123' / '456', data)

        training_file = root / '123' / '456' / 'training_combined.txt'
        assert training_file.exists()


def test_parse_topic_syntax_2():
    data = """
howdy
yo

@topic_out=blah
hi
hello
"""
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        tf.write_training_data_to_disk_v1(root / '123' / '456', data)

        training_file = root / '123' / '456' / 'training_combined.txt'
        assert training_file.exists()


def test_parse_bad_topic_no_blank_line():
    data = """
howdy
yo
@topic_out=blah
hi
hello
"""
    with pytest.raises(common.TrainingSyntaxError):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tf.write_training_data_to_disk_v1(root / '123' / '456', data)


def create_random_ai(training_root, dev_id):
    ai_id = str(uuid.uuid1())
    data = ''
    for ii in range(10):
        line = str(uuid.uuid1())
        data += line + '\n'
    root = Path(training_root)
    tf.write_training_data_to_disk_v1(root / dev_id / ai_id, data)
    return ai_id


def test_clean_ai_1():
    dev_id = str(uuid.uuid1())
    with tempfile.TemporaryDirectory() as tempdir:
        ai_id1 = create_random_ai(tempdir, dev_id)
        ai_id2 = create_random_ai(tempdir, dev_id)
        ai_id3 = create_random_ai(tempdir, dev_id)

        ai_list = tf.find_training_from_directory(tempdir)
        assert len(ai_list) == 3
        tf.delete_ai_files(tempdir, dev_id, ai_id1)

        ai_list = tf.find_training_from_directory(tempdir)
        assert len(ai_list) == 2
        assert not (dev_id, ai_id1) in ai_list


def test_upload_training_doesnt_delete_files():
    data = """
hi
hello

howdy
partner
"""
    with tempfile.TemporaryDirectory() as tempdir:
        root = Path(tempdir)
        ai_path = root / '123' / '456'
        ai_path.mkdir(parents=True)
        extra_file = ai_path / 'extra_training_file.txt'
        with extra_file.open("w") as file_handle:
            file_handle.write("Test\nTest\nTest\n")
        tf.write_training_data_to_disk_v1(ai_path, data)

        training_file = ai_path / 'training_combined.txt'
        assert training_file.exists()
        assert extra_file.exists()


if __name__ == "__main__":
    pytest.main(args=['test_training_file.py::test_clean_ai_1', '-s'])
    # pytest src/tests/test_wnet.py::test_multiprocess_3
