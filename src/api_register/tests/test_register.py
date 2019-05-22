"""Tests for api register"""
# pylint: skip-file
# flake8: noqa

import asyncio
import concurrent.futures
import json
import os
import time
import typing

import ai_training as ait
import api_register as api_r
import aiohttp
import pytest
from aiohttp import web


def _get_logger():
    logger = logging.getLogger('hu.api_register.test')
    return logger


# Test the API server calls as well, declaring a test server below as simply as possible
class MockApiServer(object):
    def __init__(self):
        self.counter = 0
        self.register_event = None
        self.fail_next_reg = False
        self.registration_data = None

    async def on_register_post(self, req):
        post_data = await req.json()
        self.registration_data = post_data
        print('POST received {}, {}'.format(req.url, post_data))
        self.counter += 1
        if self.fail_next_reg:
            self.fail_next_reg = False
            raise web.HTTPInternalServerError()

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

    async def on_get(self, req):
        print('get received')
        resp = web.Response()
        return resp

    def set_register_event(self, event):
        self.register_event = event

    def set_fail_next_reg(self):
        self.fail_next_reg = True


class MockProvider(api_r.AiStatusProviderABC):
    def __init__(self):
        self.ai_statuses = []

    async def get_ai_statuses_for_api(self):
        """Get a mapping of ai status"""
        return self.ai_statuses


@pytest.fixture
def mock_api():
    mock_api = MockApiServer()
    return mock_api


@pytest.fixture
def mock_server(mock_api):
    mock_server_app = web.Application()
    mock_server_app.router.add_post('/aiservices/register',
                                    mock_api.on_register_post)
    # make sure that the server is alive with a simpler endpoint we can use CURL against if necessary
    mock_server_app.router.add_get('/hi', mock_api.on_get)
    mock_server = aiohttp.test_utils.TestServer(mock_server_app)
    return mock_server


@pytest.fixture
def start_mock_server(loop, mock_server):
    # start pumping events
    yield loop.run_until_complete(mock_server.start_server(loop))
    ### TEARDOWN CODE ###
    loop.run_until_complete(mock_server.close())


@pytest.fixture
def mock_provider():
    mock_provider = MockProvider()
    return mock_provider


class ApiRegisterHelper:
    """Helper class to keep the tests short"""

    def __init__(self, api, provider, server):
        self.api = api
        self.provider = provider
        self.server = server
        self.ai_engine_type = 'MockBackend'
        self.this_service_url = "127.0.0.1:1234"
        self.language = "it"
        self.version = "experimental"


@pytest.fixture
def register_helper(mock_api, mock_provider, mock_server,
                    start_mock_server):
    helper = ApiRegisterHelper(mock_api, mock_provider, mock_server)
    return helper


def create_register_object(register_helper, training_enabled=True, chat_enabled=True):
    if register_helper.server is not None:
        server_url = register_helper.server.make_url('/')
        server_url = str(server_url).strip('/')
    else:
        server_url = "http://no_server_at_this_address:999999"
    reg = api_r.Register(
        training_enabled, chat_enabled,
        ai_engine_type=register_helper.ai_engine_type, 
        language=register_helper.language,
        version=register_helper.version,
        this_service_url=register_helper.this_service_url,
        provider=register_helper.provider)
    reg.api_endpoint = server_url
    return reg


async def test_register_ok(register_helper, mock_api):
    reg = create_register_object(register_helper)
    async with aiohttp.ClientSession() as http_session:
        register_ok = await reg.register(http_session)
        assert register_ok is True
        assert reg.session_id == "123456789"
        reg_data = mock_api.registration_data
        assert reg_data is not None
        assert reg_data["server_type"] == "MockBackend"
        assert reg_data["chat_capacity"] == 1
        assert reg_data["training_capacity"] == 1
        assert reg_data["language"] == "it"
        assert reg_data["version"] == "experimental"


async def test_register_chat_only(register_helper, mock_api):
    reg = create_register_object(register_helper, chat_enabled=True, training_enabled=False)
    async with aiohttp.ClientSession() as http_session:
        register_ok = await reg.register(http_session)
        assert register_ok is True
        assert reg.session_id == "123456789"
        reg_data = mock_api.registration_data
        assert reg_data is not None
        assert reg_data["server_type"] == "MockBackend"
        assert reg_data["chat_capacity"] == 1
        assert reg_data["training_capacity"] == 0

async def test_register_train_only(register_helper, mock_api):
    reg = create_register_object(register_helper, chat_enabled=False, training_enabled=True)
    async with aiohttp.ClientSession() as http_session:
        register_ok = await reg.register(http_session)
        assert register_ok is True
        assert reg.session_id == "123456789"
        reg_data = mock_api.registration_data
        assert reg_data is not None
        assert reg_data["server_type"] == "MockBackend"
        assert reg_data["chat_capacity"] == 0
        assert reg_data["training_capacity"] == 1

async def test_register_fail_server500(register_helper):
    reg = create_register_object(register_helper)
    register_helper.api.set_fail_next_reg()
    async with aiohttp.ClientSession() as http_session:
        register_ok = await reg.register(http_session)
        assert register_ok is False


async def test_register_fail_no_server(mock_api, mock_provider):
    # DON'T start the mock_server
    register_helper = ApiRegisterHelper(mock_api, mock_provider, None)

    reg = create_register_object(register_helper)
    async with aiohttp.ClientSession() as http_session:
        register_ok = await reg.register(http_session)
        assert register_ok is False


async def test_register_loop(register_helper):
    reg = create_register_object(register_helper)
    event = asyncio.Event()
    register_helper.api.set_register_event(event)
    async with aiohttp.ClientSession() as http_session:
        tasks = [
            asyncio.ensure_future(
                reg.registration_loop(http_session)),
            asyncio.ensure_future(
                event.wait())
        ]
        await asyncio.wait(tasks)
        assert reg.session_id == "123456789"


async def test_register_loop_fail_1(register_helper):
    reg = create_register_object(register_helper)
    event = asyncio.Event()
    api = register_helper.api
    api.set_fail_next_reg()
    api.set_register_event(event)

    async with aiohttp.ClientSession() as http_session:
        tasks = [
            asyncio.ensure_future(
                reg.registration_loop(http_session)),
            asyncio.ensure_future(
                event.wait())
        ]
        await asyncio.wait(tasks)
        assert reg.session_id == "123456789"


# allow debugging in VS code
if __name__ == "__main__":
    pytest.main(args=['test_register.py', '-s'])
