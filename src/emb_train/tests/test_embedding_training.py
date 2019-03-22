# flake8: noqa

import pytest
import emb_train
import emb_train.training_process
import numpy
import tempfile
from pathlib import Path
import unittest

import ai_training as ait

pytestmark = pytest.mark.asyncio

async def mock_w2v_call(payload, endpoint='words'):
    if endpoint == "words":
        return {'vectors': {"word1": [1.1, 1.2, 1.3], "word2": [0.1, 0.2, 0.3]}}
    elif endpoint == "unk_words":
        return {'unk_words': ["word1", "word2"]}
    return {}


async def get_from_er_server(relative_url, params=None):
    if relative_url == "ner":
        return [{
            'category': 'sys.places',
            'value': 'London',
            'start': 0,
            'end': 6
        }, {
            'category': 'sys.date',
            'value': 'today',
            'start': 10,
            'end': 17
        }]
    elif relative_url == "tokenize":
        return ['be', 'here', 'now']
    else:
        return {}


@pytest.fixture
async def mocked_train(mocker, loop):
    training = emb_train.training_process.EmbedTrainingProcessWorker(
        None, "no_aiohttp_session")

    mocker.patch.object(
        training.w2v_client, "w2v_call", new=mock_w2v_call)

    mocker.patch.object(
        training.entity_wrapper,
        "get_from_er_server",
        new=get_from_er_server)

    training.entity_wrapper.train_entities_q = [
        [{
            'category': 'sys.places',
            'value': 'london',
            'start': 0,
            'end': 6
        }, {
            'category': 'sys.date',
            'value': 'today',
            'start': 10,
            'end': 17
        }],
        [{
            'category': 'sys.places',
            'value': 'paris',
            'start': 0,
            'end': 5
        }, {
            'category': 'sys.person',
            'value': 'fred bloggs',
            'start': 8,
            'end': 18
        }]]

    training.entity_wrapper.train_entities_a = [
        [{
            'category': 'sys.places',
            'value': 'london',
            'start': 0,
            'end': 6
        }, {
            'category': 'sys.date',
            'value': 'today',
            'start': 10,
            'end': 17
        }],
        [{
            'category': 'sys.places',
            'value': 'paris',
            'start': 0,
            'end': 5
        }, {
            'category': 'sys.person',
            'value': 'fred bloggs',
            'start': 8,
            'end': 18
        }]]

    training.entity_wrapper.train_labels = ["You said London today",
                                            "You said Paris Fred Bloggs"]
    
    return training


def test_mocks_ok(mocked_train):
    pass


async def test_train_get_vectors(mocked_train):
    train_data = ["this is mocked", "by the function above"]
    vectors = await mocked_train.get_vectors(train_data)
    word1vec = vectors["word1"]
    word2vec = vectors["word2"]
    assert type(word1vec) is numpy.ndarray
    assert type(word2vec) is numpy.ndarray


async def test_er_entities(mocked_train):
    question = "this is a dummy question that will be mocked out"
    entities = await mocked_train.entity_wrapper.extract_entities(question)
    assert len(entities) == 2
    assert entities[0]['value'] == 'london'
    assert entities[1]['value'] == 'today'


async def test_er_tokenize(mocked_train):
    question = "this is a dummy question that will be mocked out"
    tokens = await mocked_train.entity_wrapper.tokenize(question)
    assert len(tokens) == 3
    assert tokens[0] == 'be'
    assert tokens[1] == 'here'


async def test_er_match_entities_none(mocked_train):
    question = "this question has no matching entities"
    entities = await get_from_er_server("ner")
    matched_label = mocked_train.entity_wrapper.match_entities(
        question, entities)
    assert len(matched_label) == 0


async def test_er_match_entities_1(mocked_train):
    question = "this question matches London"
    entities = await get_from_er_server("ner")
    matched_label = mocked_train.entity_wrapper.match_entities(
        question, entities)
    assert len(matched_label) == 1
    assert matched_label[0][1] == "You said London today"


async def test_er_match_entities_2(mocked_train):
    question = "this question matches Bloggs Fred"
    entities = await get_from_er_server("ner")
    matched_label = mocked_train.entity_wrapper.match_entities(
        question, entities)
    assert len(matched_label) == 1
    assert matched_label[0][1] == "You said Paris Fred Bloggs"


async def test_train_success(mocked_train, mocker):
    DUMMY_AIID = "123456"
    DUMMY_TRAINING_DATA = """
hi
hihi"""
    # mock out the maths/save functions so we can UT train()
    mocker.patch("emb_common.text_classifier_class.EmbeddingComparison.fit")
    mocker.patch("emb_common.text_classifier_class.EmbeddingComparison.save_model")
    mocker.patch("shutil.move")

    with tempfile.TemporaryDirectory() as tempdir:
        ai_path = Path(tempdir)
        train_file = ai_path / ait.AI_TRAINING_STANDARD_FILE_NAME
        with train_file.open("w") as file_handle:
            file_handle.write(DUMMY_TRAINING_DATA)
        
        msg = ait.training_process.TrainingMessage(ai_path, DUMMY_AIID, 0)
        topic = None
        await mocked_train.train(msg, topic, None)


class MockCallback:
    pass


async def dummy_async():
    pass


async def test_train_success_with_callback(mocked_train, mocker):
    DUMMY_AIID = "123456"
    DUMMY_TRAINING_DATA = """
hi
hihi"""
    # mock out the maths/save functions so we can UT train()
    mocker.patch("emb_common.text_classifier_class.EmbeddingComparison.fit")
    mocker.patch("emb_common.text_classifier_class.EmbeddingComparison.save_model")
    mocker.patch("shutil.move")
    callback = MockCallback()
    mocker.patch.object(callback, "wait_to_save", create=True, new=dummy_async)
    mocker.patch.object(callback, "report_progress", create=True)
    mocker.patch.object(callback, "check_for_cancel", create=True)

    with tempfile.TemporaryDirectory() as tempdir:
        ai_path = Path(tempdir)
        train_file = ai_path / ait.AI_TRAINING_STANDARD_FILE_NAME
        with train_file.open("w") as file_handle:
            file_handle.write(DUMMY_TRAINING_DATA)
        
        msg = ait.training_process.TrainingMessage(ai_path, DUMMY_AIID, 0)
        topic = None
        await mocked_train.train(msg, topic, callback)