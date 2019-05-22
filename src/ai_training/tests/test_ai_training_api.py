"""Tests for ai training"""
# flake8: noqa

import asyncio
import io
import json
import logging
from pathlib import Path

import ai_training as ait

import aiohttp
import pytest
from aiohttp import web

from .conftest import TRAINING_FILE_PATH


def _get_logger():
    logger = logging.getLogger('hu.ai_training.test_api')
    return logger


# Test the API server calls as well, declaring a test server below as simply as possible
class MockApiServer:
    def __init__(self):
        self.counter = 0
        self.reg_counter = 0
        self.affinity_counter = 0
        self.register_event = None
        self.registration_data = None
        self.status_event = None
        self.affinity_event = None
        self.last_status_post = None
        self.last_affinity_data = None
        self.update_error_code = None
        self.completed_error_code = None
        self.logger = _get_logger()

    async def on_status_post(self, req):
        post_data = await req.json()
        self.logger.info('POST received {}, {}'.format(req.url, post_data))
        print('POST received {}, {}'.format(req.url, post_data))
        self.last_status_post = post_data
        self.counter += 1
        await asyncio.sleep(0.01)
        if self.status_event is not None:
            self.status_event.set()
            self.status_event = None
        if self.update_error_code is not None:
            error = web.HTTPException
            error.status_code = self.update_error_code
            raise error
        if self.completed_error_code is not None:
            if post_data["training_status"] == "ai_training_complete":
                error = web.HTTPException
                error.status_code = self.completed_error_code
                raise error
        resp = web.Response()
        return resp

    async def on_register_post(self, req):
        post_data = await req.json()
        self.registration_data = post_data
        self.logger.info('POST received {}, {}'.format(req.url, post_data))
        self.reg_counter += 1

        data = {
            'server_session_id': '123456789',
            'status': {
                'code': 200,
                'info': 'registered'
            }
        }
        resp = web.json_response(data)
        if self.register_event is not None:
            self.register_event.set()
        return resp

    async def on_affinity_post(self, req):
        post_data = await req.json()
        print('POST received {}, {}'.format(req.url, post_data))
        self.last_affinity_data = post_data
        self.affinity_counter += 1
        if self.affinity_event is not None:
            self.affinity_event.set()
        resp = web.Response()
        return resp

    async def on_get(self, req):
        resp = web.Response()
        return resp

    def set_register_event(self, event):
        self.register_event = event

    def set_status_event(self, event):
        self.status_event = event

    def set_affinity_event(self, event):
        self.affinity_event = event

    def set_fail_update(self, error_code):
        self.update_error_code = error_code

    def set_completed_fail_update(self, error_code):
        self.completed_error_code = error_code

@pytest.fixture
async def test_client_manual(mock_training):
    main_app = web.Application()
    ait.initialize_ai_training_http(main_app, mock_training)
    test_server = aiohttp.test_utils.TestServer(main_app)
    client = aiohttp.test_utils.TestClient(test_server)
    return client


@pytest.fixture
def mock_api():
    mock_api = MockApiServer()
    return mock_api


@pytest.fixture
async def mock_server(mock_api):
    mock_server_app = web.Application()
    mock_server_app.router.add_post('/aiservices/{ai_id}/status',
                                    mock_api.on_status_post)
    mock_server_app.router.add_post('/aiservices/register',
                                    mock_api.on_register_post)
    mock_server_app.router.add_post('/aiservices/affinity',
                                    mock_api.on_affinity_post)
    # make sure that the server is alive with a simpler endpoint we can use
    # CURL against if necessary
    mock_server_app.router.add_get('/hi', mock_api.on_get)
    mock_server = aiohttp.test_utils.TestServer(mock_server_app)
    return mock_server


@pytest.fixture
def loop_start_servers(loop, test_client_manual, mock_server, mock_training):
    # This is the execution loop of each test
    # Start the mock server and test client
    tasks = [
        asyncio.ensure_future(mock_server.start_server()),
        asyncio.ensure_future(test_client_manual.start_server())
    ]
    # start pumping events on both client and server
    yield loop.run_until_complete(asyncio.gather(*tasks))
    # *** TEARDOWN CODE ***
    loop.run_until_complete(test_client_manual.close())
    loop.run_until_complete(mock_server.close())

class TrainingHttpRequestHelper(object):
    """Helper class to keep the tests short"""

    def __init__(self, cli, mock_server, mock_training, mock_api):
        self.dev_id = None
        self.ai_id = None
        self.parameters = dict()
        # the cli and mock_server injected
        self.cli = cli
        self.mock_server = mock_server
        self.mock_training = mock_training
        self.mock_api = mock_api


@pytest.fixture
async def testrequest(test_client_manual, loop_start_servers, mock_server,
                mock_training, mock_api):
    """
    Declare a new fixture which take the cli fixture and expands on it
    """
    # get the root path of the started mock server
    api_root = mock_server.make_url('/')
    # pass it to the mock_training (a bit like a configuration being read)
    await mock_training.set_api_server(api_root)

    req = TrainingHttpRequestHelper(test_client_manual, mock_server,
                                    mock_training, mock_api)

    return req


async def wait_for_new_training_state(testrequest, dev_id, ai_id,
                                      current_state):
    ai_url = '/ai/{}/{}'.format(dev_id, ai_id)
    start_state_name = current_state.name
    status = start_state_name
    while status == start_state_name:
        resp = await testrequest.cli.get(ai_url)
        assert resp.status == 200
        json_data = await resp.json()
        status = json_data['status']
        if status != start_state_name:
            return status

        print('Waiting for status {}, got {}'.format(current_state.name,
                                                     status))
        event = asyncio.Event()
        testrequest.mock_api.set_status_event(event)
        await asyncio.wait([event.wait()])


async def test_status_update_1(testrequest,
                               dev_id='devpost1',
                               ai_id='aipost1',
                               training_data_str=None):
    # can't inject CLI directly as we want to run CLI and Mock Server concurrently
    # so we inject cli_coro and rename it locally to cli here
    cli = testrequest.cli

    if training_data_str is None:
        training_data_filelike = open(TRAINING_FILE_PATH)
    else:
        training_data_filelike = io.StringIO(training_data_str)

    with aiohttp.MultipartWriter('training') as mpwriter:
        mpwriter.append_json({'dev_id': dev_id, 'ai_id': ai_id})
        payload = aiohttp.payload.TextIOPayload(training_data_filelike)
        payload.set_content_disposition('attachment', filename='training.txt')
        mpwriter.append_payload(payload)

    async with cli.post('/ai', data=mpwriter) as resp:
        assert resp.status == 200
        json_data = await resp.json()
    assert json_data['status'] == 'ai_ready_to_train'
    assert 'url' in json_data
    # check data is written in correctly
    assert testrequest.mock_training.training_list[(
        dev_id, ai_id)].status.state == ait.AiTrainingState.ai_ready_to_train

    async with cli.post('/ai/{}/{}?command=start'.format(dev_id,
                                                         ai_id)) as resp:
        assert resp.status == 200

    status = await wait_for_new_training_state(
        testrequest, dev_id, ai_id, ait.AiTrainingState.ai_ready_to_train)
    assert (status == ait.AiTrainingState.ai_training.name)
    assert testrequest.mock_api.counter > 0
    status_data = testrequest.mock_api.last_status_post
    assert status_data['version'] is None
    assert status_data['language'] == "en"
    return status


async def test_training_ok(testrequest):
    # can't inject CLI directly as we want to run CLI and Mock Server concurrently
    # so we inject cli_coro and rename it locally to cli here
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    await test_status_update_1(testrequest, dev_id, ai_id)

    status = await wait_for_new_training_state(testrequest, dev_id, ai_id,
                                               ait.AiTrainingState.ai_training)
    assert status == ait.AiTrainingState.ai_training_complete.name

    # check files are written
    rootdir = Path(testrequest.mock_training.training_root_dir.name)
    ai_dir = rootdir / dev_id / ai_id
    assert ai_dir.exists() is True
    assert (ai_dir / '1.txt').exists() is True
    assert (ai_dir / '2.txt').exists() is True


async def test_training_blank_ok(testrequest):
    # can't inject CLI directly as we want to run CLI and Mock Server concurrently
    # so we inject cli_coro and rename it locally to cli here
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    await test_status_update_1(testrequest, dev_id, ai_id, "")

    status = await wait_for_new_training_state(testrequest, dev_id, ai_id,
                                               ait.AiTrainingState.ai_training)
    assert status == ait.AiTrainingState.ai_training_complete.name

    # check files are written
    rootdir = Path(testrequest.mock_training.training_root_dir.name)
    ai_dir = rootdir / dev_id / ai_id
    assert ai_dir.exists() is True


async def test_chat_after_training_blank_ok(testrequest):
    # can't inject CLI directly as we want to run CLI and Mock Server concurrently
    # so we inject cli_coro and rename it locally to cli here
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    await test_status_update_1(testrequest, dev_id, ai_id, "")

    status = await wait_for_new_training_state(testrequest, dev_id, ai_id,
                                               ait.AiTrainingState.ai_training)
    assert status == ait.AiTrainingState.ai_training_complete.name

    # check files are written
    rootdir = Path(testrequest.mock_training.training_root_dir.name)
    ai_dir = rootdir / dev_id / ai_id
    assert ai_dir.exists() is True

    cli = testrequest.cli
    async with cli.get('/ai/{}/{}/chat?q=hi'.format(dev_id, ai_id)) as resp:
        assert resp.status == 200
        json_data = await resp.json()
        assert json_data['topic_out'] is None
        assert json_data['score'] == 0.0
        assert json_data['answer'] is None


async def test_training_cancel(testrequest):
    # can't inject CLI directly as we want to run CLI and Mock Server concurrently
    # so we inject cli_coro and rename it locally to cli here
    cli = testrequest.cli
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    await test_status_update_1(testrequest, dev_id, ai_id)

    async with cli.post('/ai/{}/{}?command=stop'.format(dev_id,
                                                        ai_id)) as resp:
        assert resp.status == 200

    _get_logger().info("Stop HTTP command returned ok")
    status = await wait_for_new_training_state(testrequest, dev_id, ai_id,
                                               ait.AiTrainingState.ai_training)
    assert status == ait.AiTrainingState.ai_training_stopped.name


async def test_training_reject_update_stops_training(testrequest):
    # can't inject CLI directly as we want to run CLI and Mock Server concurrently
    # so we inject cli_coro and rename it locally to cli here
    testrequest.cli
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    await test_status_update_1(testrequest, dev_id, ai_id)

    testrequest.mock_api.set_fail_update(409)

    status = await wait_for_new_training_state(testrequest, dev_id, ai_id,
                                               ait.AiTrainingState.ai_training)
    assert status == ait.AiTrainingState.ai_training_stopped.name

async def test_training_reject_completed_ok(testrequest):
    # can't inject CLI directly as we want to run CLI and Mock Server concurrently
    # so we inject cli_coro and rename it locally to cli here
    testrequest.cli
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    await test_status_update_1(testrequest, dev_id, ai_id)
    testrequest.mock_api.set_completed_fail_update(409)


    status = await wait_for_new_training_state(testrequest, dev_id, ai_id,
                                               ait.AiTrainingState.ai_training)
    assert status == ait.AiTrainingState.ai_training_complete.name

async def test_training_reject_completed_start_ok(testrequest):
    # can't inject CLI directly as we want to run CLI and Mock Server concurrently
    # so we inject cli_coro and rename it locally to cli here
    await test_training_reject_completed_ok(testrequest)
    
    cli = testrequest.cli
    dev_id = 'devpost1'
    ai_id = 'aipost1'

    testrequest.mock_api.set_completed_fail_update(None)

    async with cli.post('/ai/{}/{}?command=start'.format(dev_id,
                                                        ai_id)) as resp:
        assert resp.status == 200
    
    status = await wait_for_new_training_state(testrequest, dev_id, ai_id,
                                               ait.AiTrainingState.ai_training)
    assert status == ait.AiTrainingState.ai_training_complete.name


async def test_wait_for_save(testrequest):
    # can't inject CLI directly as we want to run CLI and Mock Server concurrently
    # so we inject cli_coro and rename it locally to cli here
    testrequest.cli
    dev_id = 'devpost1'
    ai_id = 'aipost1'
    await test_status_update_1(testrequest, dev_id, ai_id)

    testrequest.mock_api.set_fail_update(404)

    event = asyncio.Event()
    testrequest.mock_api.set_status_event(event)
    await asyncio.wait([event.wait()])

    await asyncio.sleep(1.0)
    ai_url = '/ai/{}/{}'.format(dev_id, ai_id)
    resp = await testrequest.cli.get(ai_url)
    assert resp.status == 200
    json_data = await resp.json()
    status = json_data['status']
    assert status == ait.AiTrainingState.ai_training.name

    # check files are NOT written
    rootdir = Path(testrequest.mock_training.training_root_dir.name)
    ai_dir = rootdir / dev_id / ai_id
    assert ai_dir.exists() is True
    assert (ai_dir / '1.txt').exists() is False
    assert (ai_dir / '2.txt').exists() is False


async def test_training_too_many_fail(testrequest):
    # we only have one training slot
    for ii in range(1):
        dev_id = 'devpost{}'.format(ii)
        ai_id = 'aipost{}'.format(ii)
        await test_status_update_1(testrequest, dev_id, ai_id)

    dev_id = 'devpost2'.format(ii)
    ai_id = 'aipost2'.format(ii)
    cli = testrequest.cli

    with aiohttp.MultipartWriter('training') as mpwriter:
        mpwriter.append_json({'dev_id': dev_id, 'ai_id': ai_id})
        payload = aiohttp.payload.TextIOPayload(open(TRAINING_FILE_PATH))
        payload.set_content_disposition('attachment', filename='training.txt')
        mpwriter.append_payload(payload)

    async with cli.post('/ai', data=mpwriter) as resp:
        assert resp.status == 200
        json_data = await resp.json()
    assert json_data['status'] == 'ai_ready_to_train'
    assert 'url' in json_data
    # check data is written in correctly
    assert testrequest.mock_training.training_list[(
        dev_id, ai_id)].status.state == ait.AiTrainingState.ai_ready_to_train

    async with cli.post('/ai/{}/{}?command=start'.format(dev_id,
                                                         ai_id)) as resp:
        # !!! THIS ONE SHOULD FAIL AS THERE ARE ALREADY AN AIs in TRAINING !!!
        assert resp.status == 429


async def test_registration_ok(testrequest):
    event = asyncio.Event()

    mock_api = testrequest.mock_api
    mock_api.set_register_event(event)

    await asyncio.wait([event.wait()])
    assert mock_api.reg_counter > 0
    reg_data = mock_api.registration_data
    assert reg_data is not None
    assert reg_data["server_type"] == "MOCK"
    assert reg_data["chat_capacity"] == 1
    assert reg_data["training_capacity"] == 1
    assert reg_data["language"] == "en"
    assert reg_data["version"] is None

async def test_heartbeat_ok(testrequest):
    await test_registration_ok(testrequest)
    cli = testrequest.cli

    headers = {'content-type': 'application/json'}
    data = {'server_session_id': '123456789'}
    json_data = json.dumps(data)
    async with cli.post(
            '/ai/heartbeat', data=json_data, headers=headers) as response:
        assert response.status == 200


async def test_heartbeat_fail_1(testrequest):
    await test_registration_ok(testrequest)
    cli = testrequest.cli

    async with cli.post('/ai/heartbeat') as response:
        assert response.status == 400


async def test_heartbeat_fail_2(testrequest):
    await test_registration_ok(testrequest)
    cli = testrequest.cli

    headers = {'content-type': 'application/json'}
    data = {'server_session_id': 'not_the_correct_session_id'}
    json_data = json.dumps(data)
    async with cli.post(
            '/ai/heartbeat', data=json_data, headers=headers) as response:
        assert response.status == 400


async def test_affinity_ok(testrequest):
    await test_registration_ok(testrequest)
    event = asyncio.Event()

    mock_api = testrequest.mock_api
    mock_api.set_affinity_event(event)

    cli = testrequest.cli
    resp = await cli.get('/ai/d5/a5/chat?q=hi')
    assert resp.status == 200
    json_data = await resp.json()
    assert json_data['topic_out'] == 'The British Weather'
    assert json_data['score'] == 0.5
    assert json_data['answer'] == 'really, hi'

    await asyncio.wait([event.wait()])
    assert mock_api.affinity_counter > 0
    last_affinity = mock_api.last_affinity_data
    assert last_affinity['version'] is None
    assert last_affinity['language'] == "en"
    assert last_affinity['server_session_id'] == '123456789'
    assert last_affinity['ai_list'] == ['a5']


async def test_affinity_two_ais(testrequest):
    testrequest.mock_training.config.chat_enabled = True
    await test_affinity_ok(testrequest)
    event = asyncio.Event()

    mock_api = testrequest.mock_api
    mock_api.set_affinity_event(event)

    cli = testrequest.cli
    resp = await cli.get('/ai/d4/a4/chat?q=hi')
    assert resp.status == 200
    await asyncio.wait([event.wait()])
    assert mock_api.affinity_counter > 1
    last_affinity = mock_api.last_affinity_data
    assert last_affinity['server_session_id'] == '123456789'
    assert 'a4' in last_affinity['ai_list']


async def test_affinity_maximum(testrequest):
    config = testrequest.mock_training.config
    config.chat_enabled = True

    await test_registration_ok(testrequest)

    cli = testrequest.cli
    mock_api = testrequest.mock_api
    for ii in range(5):
        event = asyncio.Event()
        mock_api.set_affinity_event(event)
        resp = await cli.get('/ai/d_ready/ai_{}/chat?q=hi'.format(ii))
        assert resp.status == 200
        await asyncio.wait([event.wait()])

    # check that we have the maximum of 3 AIs in the list, and that these are the latest ones
    last_affinity = mock_api.last_affinity_data
    ai_list = last_affinity['ai_list']
    num_ais = len(ai_list)
    assert num_ais == 1
    assert 'ai_4' in ai_list


async def test_reregister(testrequest):
    # first registration
    await test_registration_ok(testrequest)
    cli = testrequest.cli

    # send heartbeat
    data = {'server_session_id': '123456789'}
    async with cli.post('/ai/heartbeat', json=data) as response:
        assert response.status == 200
    # wait for it to expire
    await test_registration_ok(testrequest)


async def test_shutdown(testrequest):
    # first registration
    await test_registration_ok(testrequest)
    cli = testrequest.cli

    # send heartbeat to trigger the reset of the watchdogs
    data = {'server_session_id': '123456789'}
    async with cli.post('/ai/heartbeat', json=data) as response:
        assert response.status == 200

    # wait for shutdown watchdog to "terminate" everything
    event = asyncio.Event()
    testrequest.mock_training.is_killed_event = event
    await asyncio.wait([event.wait()])
    assert testrequest.mock_training.is_killed


# allow debugging in VS code
if __name__ == "__main__":
    pytest.main(args=['test_ai_training_api.py::test_training_ok', '-s'])
    # pytest src/tests/test_wnet.py::test_multiprocess_3
