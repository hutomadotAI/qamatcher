"""Tests for ai training"""
# flake8: noqa
import io
import asyncio

import ai_training as ait

import aiohttp
import pytest
from aiohttp import web

from .conftest import TRAINING_FILE_PATH

pytestmark = pytest.mark.asyncio

@pytest.fixture
def cli(aiohttp_client, mock_training):
    """
    This fixture s called "cli" in the aiohttp help, re-use the same name
    This "runs" the server
    """
    app = web.Application()
    ait.initialize_ai_training_http(app, mock_training)
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(aiohttp_client(app))


async def upload_training(cli, dev_id, ai_id, mock_training):
    with aiohttp.MultipartWriter('training') as mpwriter:
        mpwriter.append_json({'dev_id': dev_id, 'ai_id': ai_id})
        payload = aiohttp.payload.TextIOPayload(open(TRAINING_FILE_PATH))
        payload.set_content_disposition('attachment', filename='training.txt')
        mpwriter.append_payload(payload)
        resp = await cli.post('/ai', data=mpwriter)

    assert resp.status == 200
    json = await resp.json()
    assert json['status'] == 'ai_ready_to_train'
    assert 'url' in json
    # check data is written in correctly
    assert mock_training.training_list[(
        dev_id, ai_id)].status.state == ait.AiTrainingState.ai_ready_to_train


async def test_get_ai(cli):
    resp = await cli.get('/ai/d1/a1')
    assert resp.status == 200
    json = await resp.json()
    assert json['dev_id'] == "d1"
    assert json['ai_id'] == "a1"
    assert json['status'] == "ai_training"


async def test_missing_ai(cli):
    resp = await cli.get('/ai/d2/a1')
    assert resp.status == 404


async def test_upload_training_good(cli, mock_training):
    await upload_training(cli, 'devpost1', 'aipost1', mock_training)


async def test_upload_blank(cli, mock_training):
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    training_file_filelike = io.StringIO("")
    with aiohttp.MultipartWriter('training') as mpwriter:
        mpwriter.append_json({'dev_id': dev_id, 'ai_id': ai_id})
        payload = aiohttp.payload.TextIOPayload(training_file_filelike)
        payload.set_content_disposition('attachment', filename='training.txt')
        mpwriter.append_payload(payload)
        resp = await cli.post('/ai', data=mpwriter)

    assert resp.status == 200
    json = await resp.json()
    assert json['status'] == 'ai_ready_to_train'
    assert 'url' in json
    # check data is written in correctly
    assert mock_training.training_list[(
        dev_id, ai_id)].status.state == ait.AiTrainingState.ai_ready_to_train


async def test_upload_training_exists(cli, mock_training):
    # try and upload to the one AI we know is already there
    dev_id = 'd1'
    ai_id = 'a1'
    with aiohttp.MultipartWriter('training') as mpwriter:
        mpwriter.append_json({'dev_id': dev_id, 'ai_id': ai_id})
        payload = aiohttp.payload.TextIOPayload(open(TRAINING_FILE_PATH))
        payload.set_content_disposition('attachment', filename='training.txt')
        mpwriter.append_payload(payload)
        resp = await cli.post('/ai', data=mpwriter)

    assert resp.status == 200
    await resp.json()


async def test_upload_training_bad_dev_id(cli):
    with aiohttp.MultipartWriter('training') as mpwriter:
        mpwriter.append_json({'ai_id': 'aipost1'})
        payload = aiohttp.payload.TextIOPayload(open(TRAINING_FILE_PATH))
        payload.set_content_disposition('attachment', filename='training.txt')
        mpwriter.append_payload(payload)
        resp = await cli.post('/ai', data=mpwriter)

    assert resp.status == 400


async def test_upload_training_bad_ai_id(cli):
    with aiohttp.MultipartWriter('training') as mpwriter:
        mpwriter.append_json({'dev_id': 'aipost1'})
        payload = aiohttp.payload.TextIOPayload(open(TRAINING_FILE_PATH))
        payload.set_content_disposition('attachment', filename='training.txt')
        mpwriter.append_payload(payload)
        resp = await cli.post('/ai', data=mpwriter)

    assert resp.status == 400


async def test_upload_training_missing_data_1(cli):
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    with aiohttp.MultipartWriter('training') as mpwriter:
        mpwriter.append_json({'dev_id': dev_id, 'ai_id': ai_id})
        resp = await cli.post('/ai', data=mpwriter)

    assert resp.status == 400


async def test_upload_training_bad_filename(cli):
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    with aiohttp.MultipartWriter('training') as mpwriter:
        mpwriter.append_json({'dev_id': dev_id, 'ai_id': ai_id})
        payload = aiohttp.payload.TextIOPayload(open(TRAINING_FILE_PATH))
        payload.set_content_disposition(
            'attachment', filename='training-WRONGNAME.txt')
        mpwriter.append_payload(payload)
        resp = await cli.post('/ai', data=mpwriter)

    assert resp.status == 400


async def test_training_start_ok(cli):
    resp = await cli.post('/ai/d2/a2?command=start')
    # this test is freezing sometimes in teardown, so pause before it ends
    await asyncio.sleep(0.1)
    assert resp.status == 200


async def test_training_over_training_capacity_fails(cli, mock_training):
    # upload training so that we can fill the 1 training slots
    await upload_training(cli, 'd2', 'a2', mock_training)
    await upload_training(cli, 'd3', 'a3', mock_training)
    resp = await cli.post('/ai/d2/a2?command=start')
    assert resp.status == 200

    resp = await cli.post('/ai/d3/a3?command=start')
    assert resp.status == 429


async def test_training_start_fail_1(cli):
    resp = await cli.post('/ai/d2/a1?command=start')
    assert resp.status == 404


async def test_training_start_already_started(cli):
    resp = await cli.post('/ai/d1/a1?command=start')
    # this test is freezing sometimes in teardown, so pause before it ends
    await asyncio.sleep(0.1)
    assert resp.status == 200
    await resp.json()


async def test_training_stop_ok(cli):
    resp = await cli.post('/ai/d1/a1?command=stop')
    assert resp.status == 200


async def test_training_stop_fail_1(cli):
    resp = await cli.post('/ai/d2/a1?command=stop')
    assert resp.status == 404


async def test_training_stop_already_stopped(cli):
    resp = await cli.post('/ai/d2/a2?command=stop')
    assert resp.status == 200


async def test_training_unknown_command_fail(cli):
    resp = await cli.post('/ai/d2/a2?command=random')
    assert resp.status == 400


async def test_delete_ai_completed(cli, mock_training):
    resp = await cli.delete('/ai/d5/a5')
    assert resp.status == 200

    # check data is written in correctly
    assert not ("d5", "a5") in mock_training.training_list


async def test_delete_ai_in_training(cli, mock_training):
    resp = await cli.delete('/ai/d1/a1')
    assert resp.status == 200

    # check data is written in correctly
    assert not ('d1', 'a1') in mock_training.training_list


async def test_delete_ai_fail(cli):
    resp = await cli.delete('/ai/d1/a2')
    assert resp.status == 404


async def test_delete_dev(cli, mock_training):
    resp = await cli.delete('/ai/d1')
    assert resp.status == 200
    assert not ('d1', 'a1') in mock_training.training_list


async def test_delete_dev_fail(cli):
    resp = await cli.delete('/ai/d1-not-there')
    assert resp.status == 404


async def test_chat_fail_no_capacity(cli, mock_training):
    mock_training.set_chat_enabled(False)
    resp = await cli.get('/ai/d5/a5/chat?q=hi')
    assert resp.status == 400


async def test_chat_ok_1(cli):
    resp = await cli.get('/ai/d5/a5/chat?q=hi')
    assert resp.status == 200
    json_data = await resp.json()
    assert json_data['topic_out'] == 'The British Weather'
    assert json_data['score'] == 0.5
    assert json_data['answer'] == 'really, hi'


async def test_chat_ok_2(cli):
    resp = await cli.get('/ai/d4/a4/chat?q=hi')
    assert resp.status == 200
    json_data = await resp.json()
    assert json_data['topic_out'] == 'The British Weather'
    assert json_data['score'] == 0.5
    assert json_data['answer'] == 'really, hi'


async def test_chat_v2_ok_1(cli):
    data = '{ "conversation" : "hi", "entities" : ""}'
    resp = await cli.post('/ai/d5/a5/chat_v2', data=data)
    assert resp.status == 200
    json_data = await resp.json()
    assert json_data['topic_out'] == 'The British Weather'
    assert json_data['score'] == 0.5
    assert json_data['answer'] == 'really, hi'


async def test_chat_v2_ok_with_entities(cli):
    data = '{ "conversation" : "hi", "entities" : { "large": [ "CakeSize", "CoffeeSize" ], "chocolate": [ "CakeType" ] }}'
    resp = await cli.post('/ai/d5/a5/chat_v2', data=data)
    assert resp.status == 200
    json_data = await resp.json()
    assert json_data['topic_out'] == '{"large": ["CakeSize", "CoffeeSize"], "chocolate": ["CakeType"]}'
    assert json_data['score'] == 0.5
    assert json_data['answer'] == 'really, hi'


async def test_chat_hash_match(cli):
    resp = await cli.get('/ai/d5/a5/chat?q=hi&ai_hash=file_hash-data_hash')
    assert resp.status == 200


async def test_chat_hash_fail(cli):
    resp = await cli.get('/ai/d5/a5/chat?q=hi&ai_hash=incorrect_hash')
    assert resp.status == 400


async def test_chat_hash_allow_None(cli):
    resp = await cli.get('/ai/d5/a5/chat?q=hi&ai_hash=None')
    assert resp.status == 200


async def test_chat_hash_allow_empty(cli):
    resp = await cli.get('/ai/d5/a5/chat?ai_hash=&q=hi')
    assert resp.status == 200


async def test_chat_fail_404(cli):
    resp = await cli.get('/ai/d2/a1/chat?q=hi')
    assert resp.status == 404


async def test_chat_fail_not_ready_1(cli):
    resp = await cli.get('/ai/d3/a3/chat?q=hi')
    assert resp.status == 400


async def test_chat_history_none(cli):
    resp = await cli.get('/ai/d5/a5/chat?q=hi')
    assert resp.status == 200
    json_data = await resp.json()
    assert json_data['history'] is None


async def test_chat_history_set(cli, mock_training):
    history_to_inject = "This is some history"
    mock_training.chat_history = history_to_inject
    resp = await cli.get('/ai/d5/a5/chat?q=hi')
    assert resp.status == 200
    json_data = await resp.json()
    assert json_data['history'] == history_to_inject


async def test_training_not_found_gives_blank(cli, mock_training):
    mock_training.raise_training_not_found = True
    resp = await cli.get('/ai/d5/a5/chat?q=hi')
    assert resp.status == 200
    json_data = await resp.json()
    assert json_data['answer'] is None


async def test_internal_error_500(cli, mock_training):
    mock_training.add_training_with_state(
        'test_error',
        'test_error',
        ait.AiTrainingStatusWithProgress(ait.AiTrainingState.ai_undefined),
        exception_on_access=True)
    resp = await cli.get('/ai/test_error/test_error/chat?q=hi')
    assert resp.status == 500
    text_data = await resp.text()
    text_data_lower = text_data.lower()
    assert "internal server error" in text_data_lower
    assert "Traceback (" in text_data
    assert "ConfTestException" in text_data


async def test_statuses(cli, mock_training):
    resp = await cli.get('/ai/statuses')
    assert resp.status == 200
    json_data = await resp.json()
    # mock training creates a certain number of test AIs
    assert len(json_data) == 27


async def test_chat_no_503_if_no_capacity_when_chatting(cli, mock_training):
    # start chat on a5
    future1 = asyncio.ensure_future(cli.get('/ai/d5/a5/chat?q=hi'))
    # immediately start chat on a4 - this should fail as chat on a5 is active and
    # takes us over capacity
    future2 = asyncio.ensure_future(cli.get('/ai/d4/a4/chat?q=hi'))
    results = await asyncio.gather(future1, future2)
    # results don't necessarily come back in order from gather
    result_statuses = [result.status for result in results]
    for status in result_statuses:
        assert 200 == status


async def test_chat_503_if_chat_lock_limit_breached(cli, mock_training):
    # start a whole set of chats on a5 in parallel
    futures = []
    for ii in range(100):
        future = asyncio.ensure_future(
            cli.get('/ai/d5/a5/chat?q=hi'))
        futures.append(future)

    # wait for the results
    results = await asyncio.gather(*futures)

    chat_fail_503 = 0
    for result in results:
        status = result.status
        if status == 200:
            continue
        elif status == 503:
            # this chat failed with 503
            chat_fail_503 += 1
        else:
            raise Exception("Unexpected status {}".format(status))

    # we expect several chat to fail with code 503
    assert chat_fail_503 > 0


async def test_crash_if_process_pool_unhealthy_1(cli, mock_training, mocker):
    # Patch is_alive to indicate our worker process died
    mock_is_alive = mocker.patch('multiprocessing.Process.is_alive')
    mock_is_alive.return_value = False
    # Patch sys.exit so we can tell if the process aborted
    mock_exit = mocker.patch('sys.exit')

    await cli.post('/ai/d2/a2?command=start')
    assert mock_exit.call_count == 1


async def test_crash_if_process_pool_unhealthy_2(cli, mock_training, mocker):
    # Patch is_alive to indicate our worker process died
    mock_is_alive = mocker.patch('multiprocessing.Process.is_alive')
    mock_is_alive.return_value = False
    # Patch sys.exit so we can tell if the process aborted
    mock_exit = mocker.patch('sys.exit')

    await cli.get('/ai/d5/a5/chat?q=hi')
    assert mock_exit.call_count == 1


# allow debugging in VS code
if __name__ == "__main__":
    pytest.main(args=['test_ai_training.py::test_training_ok', '-s'])
    # pytest src/tests/test_wnet.py::test_multiprocess_3
