import tempfile
from pathlib import Path

import ai_training
import pytest


def test_single_QnA_file():
    """The training interface can be used as file or string, this is the only test from file"""
    data = \
        """hi
        hello
        """

    with tempfile.TemporaryDirectory() as tempdir:
        tmp_path = Path(tempdir)/"train.txt"
        with tmp_path.open(mode='w+') as tmp:
            tmp.write(data)

        topic = ai_training.file_load_training_data_v1(tmp_path)
        assert topic.topic_name is None
        assert len(topic.sub_topics) == 0
        assert len(topic.entries) == 1
        assert topic.entries[0].question == 'hi'
        assert topic.entries[0].answer == 'hello'


def test_single_QnA_str():
    data = \
        """hi
        hello
        """
    topic = ai_training.str_load_training_data_v1(data)
    assert topic.topic_name is None
    assert len(topic.sub_topics) == 0
    assert len(topic.entries) == 1
    assert topic.entries[0].question == 'hi'
    assert topic.entries[0].answer == 'hello'


def test_pass_blank():
    data = \
        """
        """
    topic = ai_training.str_load_training_data_v1(data)
    assert topic.is_empty()


def test_fail_no_answer():
    data = \
        """
        hi
        """
    with pytest.raises(ai_training.TrainingSyntaxError):
        ai_training.str_load_training_data_v1(data)


def test_single_multiturn():
    data = \
        """hi
        hello
        whazzup
        how's it going
        what's the weather like
        it's raining
        where are u
        here
"""
    topic = ai_training.str_load_training_data_v1(data)
    assert topic is not None
    assert topic.topic_name is None
    assert len(topic.sub_topics) == 0
    assert len(topic.entries) == 4
    assert topic.entries[3].question == 'where are u'
    assert topic.entries[3].answer == 'here'
    assert topic.entries[3].multi_turn_previous == "it's raining"


def test_two_QnA():
    data = \
        """hi
        hello

        whazzup
        how's it going
        """
    topic = ai_training.str_load_training_data_v1(data)
    assert topic is not None
    assert topic.topic_name is None
    assert len(topic.sub_topics) == 0
    assert len(topic.entries) == 2
    assert topic.entries[1].question == 'whazzup'
    assert topic.entries[1].answer == "how's it going"


def test_several_QnA():
    data = \
        """
        hello
        hi
        how are you
        i am fine, you?

        how are you
        not too bad

        good morning
        hello"""
    topic = ai_training.str_load_training_data_v1(data)
    assert topic is not None
    assert topic.topic_name is None
    assert len(topic.sub_topics) == 0
    assert len(topic.entries) == 4
    assert topic.entries[0].multi_turn_previous is None
    assert topic.entries[1].question == 'how are you'
    assert topic.entries[1].answer == "i am fine, you?"
    assert topic.entries[1].multi_turn_previous == "hi"


def test_topic_one():
    data = \
        """
        hello
        hi

        @topic_out=friendly
        how are you
        i am fine, you?
        """
    topic = ai_training.str_load_training_data_v1(data)
    assert topic is not None
    assert topic.topic_name is None
    assert len(topic.sub_topics) == 1

    # root topic needs to have the first entry from the topic
    topic_hash = ai_training.hash_topic_name("friendly")
    assert len(topic.entries) == 2
    assert topic.entries[0].question == 'hello'
    assert topic.entries[0].answer == "hi"
    assert topic.entries[1].question == "how are you"
    assert topic.entries[1].answer == "i am fine, you?"
    assert topic.entries[1].topic_out == topic_hash
    assert topic.sub_topics[topic_hash].entries[0].question == "how are you"
    assert topic.sub_topics[topic_hash].entries[0].answer == "i am fine, you?"
    assert topic.sub_topics[topic_hash].entries[0].topic_out is None


def test_topic_one_reordered():
    data = \
        """
        @topic_out=friendly
        how are you
        i am fine, you?

        hello
        hi
        """
    topic = ai_training.str_load_training_data_v1(data)
    assert topic is not None
    assert topic.topic_name is None
    assert len(topic.sub_topics) == 1

    # root topic needs to have the first entry from the topic
    topic_hash = ai_training.hash_topic_name("friendly")
    assert len(topic.entries) == 2
    assert topic.entries[0].question == 'hello'
    assert topic.entries[0].answer == "hi"
    assert topic.sub_topics[topic_hash].entries[0].question == "how are you"
    assert topic.sub_topics[topic_hash].entries[0].answer == "i am fine, you?"


def test_fail_topic_no_root():
    data = \
        """
        @topic_out=friendly
        how are you
        i am fine, you?
        """
    with pytest.raises(ai_training.TrainingSyntaxError):
        ai_training.str_load_training_data_v1(data)


def test_topics_split():
    data = \
        """
        @topic_out=friendly
        how are you
        i am fine, you?

        hello
        hi

        @topic_out=friendly
        comment allez vous?
        tres bien?

        yo
        yo to you too
        """
    topic = ai_training.str_load_training_data_v1(data)
    assert topic is not None
    assert topic.topic_name is None
    assert len(topic.sub_topics) == 1
    # root topic needs to have the first entry from the topic
    assert len(topic.entries) == 3
    assert topic.entries[0].question == 'hello'
    assert topic.entries[0].answer == "hi"
    topic_hash = ai_training.hash_topic_name("friendly")
    assert len(topic.sub_topics[topic_hash].entries) == 2
    assert topic.sub_topics[topic_hash].entries[0].question == "how are you"
    assert topic.sub_topics[topic_hash].entries[0].answer == "i am fine, you?"


def test_parse_bad_topic_no_blank_line():
    data = """
howdy
yo
@topic_out=blah
hi
hello
"""
    with pytest.raises(ai_training.TrainingSyntaxError):
        ai_training.str_load_training_data_v1(data)
