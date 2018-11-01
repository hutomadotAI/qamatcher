# flake8: noqa

import pytest
import tempfile

import embedding.chat_process

import ai_training.chat_process as ait_c


async def mock_w2v_call(payload):
    return {'vectors': {"word1": [1.1, 1.2, 1.3], "word2": [0.1, 0.2, 0.3]}}


async def mock_unk_words_w2v_call(payload):
    return {'unk_words':
            ['this', 'be', 'question', 'for', 'embedding',
             'shall', 'with', 'match', 'how', 'you']}


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
        if params['q'] == "This is London today for entity match":
            return ["this", "be", "london", "today", "for", "entity", "match"]
        elif params['q'] == "This is a perfect string match":
            return ["this", "be", "perfect", "string", "match"]
        elif params['q'] == "This is the question for embedding word1 word2":
            return ["this", "be", "question", "for", "embedding", "word1", "word2"]
        elif params['q'] == "This should match with word1 word2":
            return ["this", "shall", "match", "with", "word1", "word2"]
        elif params['q'] == "How are you?":
            return ["UNK"]
        elif params['q'] == "This is a @{custom_ent}@ match":
            return ["this", "be", "@{custom_ent}@", "match"]
        elif params['q'] == "@{week}@":
            return ["@{week}@"]
        else:
            return ["UNK"]
    else:
        return {}


@pytest.fixture
async def mocked_chat(mocker, loop):
    chat = embedding.chat_process.EmbeddingChatProcessWorker(
        None, loop, "no_aiohttp_session")

    mocker.patch.object(
        chat.w2v_client, "get_vectors_for_words", new=mock_w2v_call)

    mocker.patch.object(
        chat.w2v_client, "get_unknown_words", new=mock_unk_words_w2v_call)

    mocker.patch.object(
        chat.entity_wrapper,
        "get_from_er_server",
        new=get_from_er_server)

    chat.entity_wrapper.train_entities_q = [
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
        [], [], [], []]

    chat.entity_wrapper.train_entities_a = [
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
        [], [], [], []]

    chat.entity_wrapper.train_labels = ["entity wins with London today",
                                        "string wins",
                                        "embedding wins",
                                        "custom entity match",
                                        "week"]

    chat.string_match.train_data = [
        ("This is London today for entity match", "entity wins with London today"),
        ("This is a perfect string match", "string wins"),
        ("This is the question for embedding word1 word2", "embedding wins"),
        ("This is a @{custom_ent}@ match", "custom entity match"),
        ("@{week}@", "week")]
    chat.string_match.tok_train = [
        ["this", "be", "london", "today", "for", "entity", "match"],
        ["this", "be", "perfect", "string", "match"],
        ["this", "be", "question", "for", "embedding", "word1", "word2"],
        ["this", "be", "@{custom_ent}@", "match"],
        ["@{week}@"]
    ]
    chat.string_match.tok_train_no_sw = [
        ["this", "be", "london", "today", "for", "entity", "match"],
        ["this", "be", "perfect", "string", "match"],
        ["this", "be", "question", "for", "embedding", "word1", "word2"],
        ["this", "be", "@{custom_ent}@", "match"],
        ["@{week}@"]
    ]
    chat.string_match.cust_ents_train = [
        [], [], [], ["custom_ent"], ["week"]
    ]
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


async def test_chat_request_string_match(mocker, mocked_chat):
    score = 1.00
    mocker.spy(mocked_chat.entity_wrapper, "match_entities")

    msg = ait_c.ChatRequestMessage("This is a perfect string match",
                                   None, None, update_state=True, entities=None)
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "string wins"
    print(response.score)
    assert round(response.score, 2) == score
    assert response.topic_out is None
    assert response.history is None
    assert mocked_chat.entity_wrapper.match_entities.call_count == 1


async def test_chat_request_entity_match(mocker, mocked_chat):
    score = float(embedding.chat_process.ENTITY_MATCH_PROBA)
    mocker.spy(mocked_chat.entity_wrapper, "match_entities")

    msg = ait_c.ChatRequestMessage("This should give an entity match for London and today",
                                   None, None, update_state=True, entities=None)
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "entity wins with London today"
    assert response.score == score
    assert response.topic_out is None
    assert response.history is None
    assert mocked_chat.entity_wrapper.match_entities.call_count == 2


async def test_chat_request_embedding_match(mocker, mocked_chat):
    score = 0.85
    mocker.spy(mocked_chat.entity_wrapper, "match_entities")
    mocker.patch("embedding.text_classifier_class.EmbeddingComparison.predict")
    embedding.text_classifier_class.EmbeddingComparison.predict.return_value = (
        ["embedding wins"], [1.0])

    msg = ait_c.ChatRequestMessage("This should match with word1 word2",
                                   None, None, update_state=True, entities=None)
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "embedding wins"
    assert response.score == score
    assert response.topic_out is None
    assert response.history is None
    assert mocked_chat.entity_wrapper.match_entities.call_count == 1


async def test_chat_request_no_match(mocker, mocked_chat):
    score = 0.0
    mocker.spy(mocked_chat.entity_wrapper, "match_entities")

    msg = ait_c.ChatRequestMessage("How are you?", None, None, update_state=True, entities=None)
    response = await mocked_chat.chat_request(msg)
    assert response.answer == ""
    assert response.score == score
    assert response.topic_out is None
    assert response.history is None
    assert mocked_chat.entity_wrapper.match_entities.call_count == 2


async def test_custom_entity_match(mocker, mocked_chat):
    score = 1.0
    msg = ait_c.ChatRequestMessage("This is a custom entity match",
                                   None, None, update_state=True,
                                   entities={"custom entity": ["custom_ent", "fake_tag"],
                                             "fake entity value": ["fake_tag2"]})
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "custom entity match"
    assert response.score == score
    assert response.topic_out is None
    assert response.history is None


async def test_casing_match(mocker, mocked_chat):
    score = 1.0
    msg = ait_c.ChatRequestMessage("week 1",
                                   None, None, update_state=True,
                                   entities={"Week 1": ["week"]})
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "week"
    assert response.score == score

    msg = ait_c.ChatRequestMessage("Week 2",
                                   None, None, update_state=True,
                                   entities={"Week 2": ["week"]})
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "week"
    assert response.score == score

    msg = ait_c.ChatRequestMessage("wEEk 3",
                                   None, None, update_state=True,
                                   entities={"wEEk 3": ["week"]})
    response = await mocked_chat.chat_request(msg)
    assert response.answer == "week"
    assert response.score == score

