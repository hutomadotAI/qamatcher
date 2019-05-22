"""
Module to show/control status of AIs via HTTP (asyncio/aiohttp)
"""
import asyncio
import datetime
import http
import json
import logging
import multiprocessing
import socket
import random
import traceback
import sys

import aiohttp
from aiohttp import web

import ai_training as ait
import ai_training.http_common as ait_http
import ai_training.http_item as http_item
import ai_training.save_controller as save_controller
import ai_training.common as aitc

import api_register
import async_process_pool.watchdog as a_watchdog
import async_process_pool.process_pool as a_pool


def _get_logger():
    logger = logging.getLogger('hu.ai_training.http')
    return logger


class ReinitializeError(aitc.Error):
    """Error fired when multiple initialization"""
    pass


class ShutdownException(aitc.Error):
    """Shutdown exception fired when watchdog timer expires"""
    pass


class HttpAiCollection(api_register.AiStatusProviderABC,
                       ait.AiTrainingControllerABC):
    """Collection of AI training object for aiohttp
       Pass in a lookup to the object to the actual training data"""

    def __init__(self, training_lookup: ait.AiTrainingProviderABC):
        self.training_lookup = training_lookup
        self.__http_client_session = None
        self.logger = _get_logger()
        self.api_register = None
        self.reregister_watchdog = None
        self.shutdown_watchdog = None
        self.chat_lock = asyncio.Lock()
        self.__registration_task = None
        self.__save_controller = None
        self.__multiprocessing_manager = multiprocessing.Manager()
        self.__chat_startup_lock = asyncio.Lock()
        self.__aiohttp_app = None
        self.__initialized = False

    async def training_callback(self, item):
        """Callback from an AI that there is status information to send
        to API"""
        config = self.training_lookup.config
        try:
            self.logger.debug("In training_callback")
            status_with_progress = item.status
            state_name = status_with_progress.state.name
            data = {
                'dev_id': item.dev_id,
                'ai_id': item.ai_id,
                'ai_engine': self.training_lookup.ai_engine_name,
                'training_status': state_name,
                'version': config.version,
                'language': config.language
            }
            if status_with_progress.training_progress is not None:
                data['training_progress'] = float(
                    status_with_progress.training_progress)
            if status_with_progress.training_error is not None:
                data['training_error'] = float(
                    status_with_progress.training_error)
            if status_with_progress.training_hash is None:
                data['ai_hash'] = None
            else:
                data['ai_hash'] = str(status_with_progress.training_hash)
            if self.api_register.session_id is not None:
                data['server_session_id'] = self.api_register.session_id

            url = 'aiservices/{}/status'.format(item.ai_id)
            self.logger.debug("Status update '{}' to {}".format(
                state_name, url))

            status = await self._send_update_to_api(url, data)

            # if we get a CONFLICT response, stop training
            if status is http.HTTPStatus.CONFLICT:
                item.training_rejected()
            elif status is None or status is http.HTTPStatus.OK:
                self.__save_controller.set_save_state(item.ai_id, True)
            else:
                self.__save_controller.set_save_state(item.ai_id, False)

        except Exception:
            # We want to log and swallow exception and not kill the
            # calling watch loop
            self.logger.error(
                "Error caught in training callback", exc_info=True)

    def start_registration_with_api(self):
        # refresh API server
        api_server = self.training_lookup.config.api_server
        self.api_register.api_endpoint = api_server
        if api_server:
            self.__registration_task = asyncio.create_task(
                self.__register_with_api())
        else:
            self.logger.warning(
                "No api_endpoint specified, skipping registration")

    def __shutdown_watchdog_fired(self):
        self.logger.warning("Shutdown watchdog fired!")
        self.reregister_watchdog.cancel()
        # this must be a non-async function but we want this to cleanup
        coro = self.__shutdown_watchdog_fired_async()
        asyncio.create_task(coro)

    async def __shutdown_watchdog_fired_async(self):
        try:
            print("Shutdown actions: shutdown")
            await self.__aiohttp_app.shutdown()
            print("Shutdown actions: cleanup")
            await self.__aiohttp_app.cleanup()
            print("Shutdown actions: exit")
        finally:
            self.training_lookup.kill_running_process()

    async def create_training_process_pool(self, training_processes: int,
                                           training_queue_size: int,
                                           worker_type: type):
        """Create a training pool"""
        training_pool = a_pool.AsyncProcessPool(
            self.__multiprocessing_manager, 'Training_pool',
            training_processes, training_queue_size, training_queue_size)
        await training_pool.initialize_processes(
            worker_type, save_controller=self.__save_controller)
        return training_pool

    @property
    def multiprocessing_manager(self):
        return self.__multiprocessing_manager

    @property
    def save_controller(self):
        return self.__save_controller

    def get_item(self, dev_id, ai_id) -> ait.AiTrainingItemABC:
        """Get item from lookup"""
        item = self.training_lookup.lookup_item(dev_id, ai_id)
        return item

    async def load_training_data_async(self, dev_id,
                                       ai_id) -> ait.AiTrainingItemABC:
        """Get item from lookup"""
        item = await self.training_lookup.load_training_data_async(
            dev_id, ai_id)
        if item is not None:
            item.controller = self
        return item

    async def delete_item(self, dev_id, ai_id) -> ait.AiTrainingItemABC:
        """Get item from lookup"""
        item = await self.training_lookup.delete_ai(dev_id, ai_id)
        return item

    async def on_post_ai(self, req: aiohttp.web.Request):
        """POST handler - upload training data"""
        dev_id, ai_id, training_data = await self._get_training_data_from_request(req)
        self.logger.info("Add training data for AI {}/{}".format(
            dev_id, ai_id))
        item = self.training_lookup.lookup_item(dev_id, ai_id)
        if item is None:
            self.logger.info(
                "AI {}/{} is not found - creating "
                "training data".format(dev_id, ai_id))
            item = self.training_lookup.create(dev_id, ai_id)
        else:
            if item.status.is_training:
                self.logger.info(
                    "AI {}/{} is training - stopping before changing "
                    "training data".format(dev_id, ai_id))
                await item.stop_training()

        item.controller = self
        hash_value = ait.training_file.write_training_data_to_disk_v1(
            item.ai_data_directory, training_data)
        status = ait.AiTrainingStatusWithProgress(
            ait.AiTrainingState.ai_ready_to_train,
            training_file_hash=hash_value)

        # we have an explicit request to change state, so we will
        # always save this
        item.reset_status(status, always_save=True)
        await item.notify_status_update()

        url = "{}/{}/{}".format(req.url, dev_id, ai_id)
        data = {'status': item.status.state.name, 'url': url}
        resp = web.json_response(data)
        return resp

    async def on_get_statuses(self, req: aiohttp.web.Request):
        """Define an endpoint that reads the statuses of all AIs
        This allows API to query a single master instance rather than
        wait for all of them"""
        if self.training_lookup.config.training_enabled:
            ai_statuses = await self.get_ai_statuses_for_api()
            ai_data = [{
                'ai_id': ai.ai_id,
                'training_status': ai.training_status,
                'ai_hash': ai.ai_hash
            } for ai in ai_statuses]
        else:
            # If no training capacity don't report AIs
            ai_data = []
        resp = web.json_response(ai_data)
        return resp

    async def on_delete_dev(self, req):
        """Request to delete a dev"""
        dev_id = req.match_info['dev_id']
        try:
            await self.training_lookup.delete_dev(dev_id)
        except ait.TrainingNotFoundError:
            raise aiohttp.web.HTTPNotFound()
        resp = web.Response()
        return resp

    async def on_startup(self, app):
        """Initialise server"""
        if self.__initialized:
            raise ReinitializeError("Initialized more than once")
        self.__initialized = True

        self.__aiohttp_app = app
        self.__save_controller = save_controller.SaveController(
            self.__multiprocessing_manager)
        self.__http_client_session = aiohttp.ClientSession()
        self.training_lookup.controller = self

        await self.training_lookup.on_startup()
        for _, value in self.training_lookup.items():
            value.controller = self

        engine = self.training_lookup.ai_engine_name
        config = self.training_lookup.config
        heartbeat_timeout = (
            config.api_heartbeat_timeout_seconds * random.uniform(0.8, 1.2))
        self.reregister_watchdog = a_watchdog.Watchdog(
            heartbeat_timeout, self.start_registration_with_api,
            "API re-register")

        shutdown_timeout = (
            config.api_shutdown_timeout_seconds * random.uniform(0.8, 1.2))
        self.shutdown_watchdog = a_watchdog.Watchdog(
            shutdown_timeout, self.__shutdown_watchdog_fired,
            "Shutdown")
        training_enabled = config.training_enabled
        chat_enabled = config.chat_enabled
        this_server_url = config.this_server_url

        self.logger.info(
            "*** Started backend server type={}, version={}, language={}".
            format(engine, config.version, config.language))
        self.logger.info(
            "*** AI root directory is at {}".
            format(config.training_data_root))
        if not this_server_url:
            # Kubernetes doesn't deal with DNS entries, so we'll need to use
            # the pod's IP which
            # will be accessible from other pods in the cluster.
            # Docker swarm should work either way.
            hostname = socket.getfqdn(
            )  # Using FQDN makes it more likely to work on dev PCs too
            primary_ip = socket.gethostbyname(hostname)
            this_server_url = "http://{}:9090/ai".format(primary_ip)
            self.logger.info(
                "*** Server URL not specified, using {}".format(
                    this_server_url))
        else:
            self.logger.info(
                "*** Service URL={}".
                format(this_server_url))
        self.logger.info("*** Enabled: train={}, chat={}".format(
            training_enabled, chat_enabled))
        self.api_register = api_register.Register(
            training_enabled,
            chat_enabled,
            ai_engine_type=engine,
            this_service_url=this_server_url,
            language=config.language,
            version=config.version,
            provider=self)
        self.start_registration_with_api()

    async def on_heartbeat(self, req):
        """Incoming heartbeat from API"""
        try:
            json_data = await req.json()
        except json.JSONDecodeError:
            ait_http.raise_bad_request('missing session ID as Json in request')

        try:
            session_id = json_data['server_session_id']
        except KeyError:
            ait_http.raise_bad_request('missing session ID in request')

        expected_session_id = self.api_register.session_id
        if session_id != expected_session_id:
            ait_http.raise_bad_request(
                'Invalid session ID in request, was expecting {}, got {}'.
                format(expected_session_id, session_id))

        self.reregister_watchdog.reset_watchdog()
        self.shutdown_watchdog.reset_watchdog()
        resp = web.Response()
        return resp

    def is_training_slot_available(self):
        """Returns true if there is an available training slot"""
        if not self.training_lookup.config.training_enabled:
            return False

        for _, value in self.training_lookup.items():
            if value.is_training_active:
                return False

        return True

    async def start_chat_for_ai(self, item: ait.AiTrainingItemABC):
        """Starts chat for a given AI"""
        config = self.training_lookup.config
        # Make sure that we can only process on chat start at a time
        async with self.__chat_startup_lock:
            chat_enabled = config.chat_enabled
            if not chat_enabled:
                ait_http.raise_bad_request("Chat not enabled on this server")

            active_chat_count = 0
            oldest_chat = None
            # create a set of AIs that are active, this is the chat affinity
            # This includes the AI we are about to start chatting to
            chat_affinity = {item.ai_id}
            item.last_chat_time = datetime.datetime.utcnow()
            # make sure this item knows we are its controller
            item.controller = self

            for _, value in self.training_lookup.items():
                if value.last_chat_time is not None:
                    active_chat_count += 1
                    chat_affinity.add(value.ai_id)
                    if (value.ai_id != item.ai_id and
                        (oldest_chat is None or
                         value.last_chat_time < oldest_chat.last_chat_time)):
                        oldest_chat = value

            if active_chat_count > 1:
                oldest_chat.last_chat_time = None
                await oldest_chat.shutdown_chat()
                chat_affinity.remove(oldest_chat.ai_id)

            # send chat affinity
            session_id = self.api_register.session_id
            if session_id is None:
                self.logger.info(
                    "Chat affinity not sent - no session ID available")
                return

            chat_affinity_list = list(chat_affinity)
            data = {
                'server_session_id': session_id,
                'ai_list': chat_affinity_list,
                'version': config.version,
                'language': config.language
            }

            self.logger.info(
                "Chat affinity update {}".format(chat_affinity_list))
            coro = self._send_update_to_api('aiservices/affinity', data)
            asyncio.create_task(coro)

    async def on_shutdown(self):
        """Shutdown client connection"""
        await self.training_lookup.on_shutdown()
        await self.__http_client_session.close()
        if (self.__registration_task is not None
                and not self.__registration_task.done()):
            self.__registration_task.cancel()

    async def get_ai_statuses_for_api(self):
        ais = []
        # force read the current AIs to a list to avoid a race condition
        # where the dictionary changes
        training_items = list(self.training_lookup.items())
        # make sure that we reload training status from disk EVERY time
        # This will add load to the disk, but will avoid Bug 3491
        for _, item in training_items:
            item_from_disk = await self.load_training_data_async(
                item.dev_id, item.ai_id)
            status = api_register.ApiAiStatus(
                item_from_disk.status.state.name, item_from_disk.ai_id,
                item_from_disk.status.training_hash)
            ais.append(status)
        return ais

    async def __register_with_api(self):
        await self.api_register.registration_loop(self.__http_client_session)
        self.reregister_watchdog.reset_watchdog()

    async def _get_training_data_from_request(self, req: aiohttp.web.Request):
        if 'multipart' not in req.content_type:
            ait_http.raise_bad_request(
                'bad request: ai creation needs multipart/form-data '
                'content type')

        try:
            reader = await req.multipart()
        except ValueError:
            self.logger.warning('Failed to read multipart', exc_info=True)
            ait_http.raise_bad_request(
                'bad request: ai creation multipart is invalid')

        dev_id = None
        ai_id = None
        training_data = None
        while True:
            part = await reader.next()
            if part is None:
                break  # all read
            if part.filename == 'training.txt':
                training_data = await part.text()
            else:
                try:
                    json_data = await part.json()
                except json.JSONDecodeError:
                    ait_http.raise_bad_request(
                        'bad request: please provide dev_id and ai_id in JSON '
                        'form in multipart')
                dev_id = json_data.get('dev_id', None)
                ai_id = json_data.get('ai_id', None)
            if (training_data is not None and ai_id is not None
                    and dev_id is not None):
                break

        self._validate_training_data(dev_id, ai_id, training_data)

        return (dev_id, ai_id, training_data)

    def _validate_training_data(self, dev_id, ai_id, training_data):
        if dev_id is None:
            ait_http.raise_bad_request(
                'bad request: need JSON data with dev_id')
        elif ai_id is None:
            ait_http.raise_bad_request(
                'bad request: need JSON data with ai_id')
        elif training_data is None:
            ait_http.raise_bad_request(
                'bad request: need multi-part with training.txt')

    async def _send_update_to_api(self, relative_url, data):
        """Send update to api server."""
        api_server = self.training_lookup.config.api_server

        # default status to None
        status = None
        if api_server is None or api_server == "":
            self.logger.debug("update to %s not sent - no API URL set",
                              relative_url)
            return status

        json_to_send = json.dumps(data)
        base_url = str(api_server)
        if base_url.endswith('/'):
            base_url = base_url[:-1]

        url = "{}/{}".format(base_url, relative_url)

        try:
            headers = {'content-type': 'application/json'}
            async with self.__http_client_session.post(
                    url, data=json_to_send, headers=headers) as response:
                status = http.HTTPStatus(response.status)

                if status is http.HTTPStatus.OK:
                    self.logger.debug("Update to {} successful".format(url))
                elif status is http.HTTPStatus.CONFLICT:
                    self.logger.warning(
                        "Update to {} rejected, aborting".format(url))
                else:
                    self.logger.warning(
                        "Update to {} failed with status {}".format(
                            url, status))
        except Exception as exc:
            self.logger.error("Update to {} failed with exception: {}".format(
                url, exc.__class__.__name__))
            status = http.HTTPStatus(http.HTTPStatus.NOT_FOUND)
        return status


async def on_startup(app):
    """Registered with app, called on startup"""
    logger = _get_logger()
    logger.info('In on_startup')
    training_collection = app['training_collection']
    await training_collection.on_startup(app)


async def on_shutdown(app):
    """Registered with app, called on shutdown"""
    logger = _get_logger()
    logger.info('In on_shutdown')
    training_collection = app['training_collection']
    await training_collection.on_shutdown()


@web.middleware
async def log_error_middleware(request, handler):
    try:
        response = await handler(request)
    except a_pool.PoolUnhealthyError:
        _get_logger().exception("PoolUnhealthyError in call")
        sys.exit(1)
        raise
    except aiohttp.web_exceptions.HTTPException:
        # assume if we're throwing this that it's already logged
        raise
    except Exception:
        _get_logger().exception("Unexpected exception in call")

        error_string = "Internal Server Error\n" + traceback.format_exc()
        raise aiohttp.web_exceptions.HTTPInternalServerError(
            text=error_string)
    return response


def initialize_ai_training_http(app: web.Application,
                                training_lookup: ait.AiTrainingProviderABC):
    """Initialize aiohttp API routes"""
    logger = _get_logger()
    logger.info('In initialize_ai_training_http')
    training_collection = HttpAiCollection(training_lookup)
    app['training_collection'] = training_collection
    # add startup and shutdown logic
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.middlewares.append(log_error_middleware)
    app.router.add_post('/ai', training_collection.on_post_ai)
    app.router.add_get(
        '/ai/statuses',
        training_collection.on_get_statuses)

    app.router.add_delete(
        '/ai/{dev_id}',
        training_collection.on_delete_dev)
    training_item = http_item.HttpAiItem(training_collection)
    app.router.add_get('/ai/{dev_id}/{ai_id}',
                       training_item.on_get)
    app.router.add_post('/ai/{dev_id}/{ai_id}',
                        training_item.on_post)
    app.router.add_delete('/ai/{dev_id}/{ai_id}',
                          training_item.on_delete)
    app.router.add_get('/ai/{dev_id}/{ai_id}/chat',
                       training_item.on_chat)
    app.router.add_post('/ai/{dev_id}/{ai_id}/chat_v2',
                        training_item.on_chat_v2)

    app.router.add_post('/ai/heartbeat', training_collection.on_heartbeat)
