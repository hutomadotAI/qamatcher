from pathlib import Path
import tempfile
import ai_training


def test_hash_1():
    with tempfile.TemporaryDirectory() as tempdir:
        tmp_path = Path(tempdir)/"train.txt"
        with tmp_path.open(mode='wb+') as tmp:
            tmp.write(b'Hello world')
        tmp_path = Path(tmp.name)
        hash_str = ai_training.calculate_training_hash(tmp_path)
        assert hash_str == 'PiWWCnnbxptnTNTsZ6csYg=='


def test_hash_long_file():
    with tempfile.TemporaryDirectory() as tempdir:
        tmp_path = Path(tempdir)/"train.txt"
        with tmp_path.open(mode='wb+') as tmp:
            LONG_FILE_SIZE = 10000000
            bytes = bytearray([ii % 256 for ii in range(LONG_FILE_SIZE)])
            tmp.write(bytes)
        hash_str = ai_training.calculate_training_hash(tmp_path)
        assert hash_str == 'U2OnKrbBB3fY1i_we0RZKg=='
