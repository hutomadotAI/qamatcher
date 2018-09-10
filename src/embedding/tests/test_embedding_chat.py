# flake8: noqa

import pytest
import tempfile

import embedding.chat_process

import ai_training.chat_process as ait_c

async def mock_w2v_call(payload):
    return {'vectors': {"word1": [1.1, 1.2, 1.3], "word2": [0.1, 0.2, 0.3]}}

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
async def mocked_chat(mocker, loop):
    chat = embedding.chat_process.EmbeddingChatProcessWorker(
        None, loop, "no_aiohttp_session")

    mocker.patch.object(
        chat.w2v_client, "w2v_call", new=mock_w2v_call)

    mocker.patch.object(
        chat.entity_wrapper,
        "get_from_er_server",
        new=get_from_er_server)

    chat.entity_wrapper.train_entities = [
        [{
            'category': 'sys.places',
            'value': 'London',
            'start': 0,
            'end': 7
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

    chat.entity_wrapper.train_labels = ["You said London today",
                                            "You said Paris Fred Bloggs"]
    chat.string_match.train_data = [
        ("This is London today for entity match", "entity wins with London today"),
        ("This is a perfect string match", "string wins"),
        ("This is the question for embedding word1 word2", "embedding wins")]
    
    # mock out the load methods
    mocker.patch("embedding.text_classifier_class.EmbeddingComparison.load_model")
    mocker.patch.object(chat.entity_wrapper, "load_data")
    mocker.patch.object(chat.string_match, "load_train_data")

    DUMMY_AIID = "123456"
    # Create a temp directory for AI path
    with tempfile.TemporaryDirectory() as tempdir:
        chat.ai_path = tempdir
        chat.ai_id = DUMMY_AIID
        yield chat
        # Clean up will happen after test


def test_mocks_ok(mocked_chat):
    pass

async def test_chat_start(mocked_chat):
    msg = ait_c.WakeChatMessage(mocked_chat.ai_path, mocked_chat.ai_id)
    await mocked_chat.start_chat(msg)


async def test_chat_request_embedding_match(mocker, mocked_chat):
    score = float(embedding.chat_process.THRESHOLD + 0.1)
    mocker.patch("embedding.text_classifier_class.EmbeddingComparison.predict")
    embedding.text_classifier_class.EmbeddingComparison.predict.return_value = (
        ["the answer"], [score])
    mocker.spy(mocked_chat.entity_wrapper, "match_entities")

    msg = ait_c.ChatRequestMessage("This is the question", None, None, update_state=True)
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "the answer"
    assert response.score == score
    assert response.topic_out is None
    assert response.history is None
    assert mocked_chat.entity_wrapper.match_entities.call_count == 1

async def test_chat_request_entity_no_match(mocker, mocked_chat):
    score = float(embedding.chat_process.THRESHOLD - 0.1)
    mocker.patch("embedding.text_classifier_class.EmbeddingComparison.predict")
    embedding.text_classifier_class.EmbeddingComparison.predict.return_value = (
        ["the answer"], [score])
    mocker.spy(mocked_chat.entity_wrapper, "match_entities")

    msg = ait_c.ChatRequestMessage("This is the question", None, None, update_state=True)
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "the answer"
    assert response.score == score
    assert response.topic_out is None
    assert response.history is None
    assert mocked_chat.entity_wrapper.match_entities.call_count == 1

async def test_chat_request_entity_no_match2(mocker, mocked_chat):
    score = float(embedding.chat_process.THRESHOLD - 0.1)
    mocker.patch("embedding.text_classifier_class.EmbeddingComparison.predict")
    embedding.text_classifier_class.EmbeddingComparison.predict.return_value = (
        ["the answer"], [score])
    mocker.spy(mocked_chat.entity_wrapper, "match_entities")

    msg = ait_c.ChatRequestMessage("This question has entities London today in it", None, None, update_state=True)
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "You said London today"
    assert response.score == embedding.chat_process.ENTITY_MATCH_PROBA
    assert response.topic_out is None
    assert response.history is None
    assert mocked_chat.entity_wrapper.match_entities.call_count == 1