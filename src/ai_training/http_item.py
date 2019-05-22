"""
Module to show/control status of a single AI via HTTP (asyncio/aiohttp)
"""
import asyncio
import json
import logging

from aiohttp import web

import ai_training as ait
import ai_training.http_common as ait_http
import ai_training.common as ait_common


def _get_logger():
    logger = logging.getLogger('hu.ai_training.http')
    return logger


class HttpAiItem:
    """Single AI training object for web"""

    def __init__(self, collection):
        self.collection = collection
        self.logger = _get_logger()

    async def on_get(self, req):
        """Getter for an AI"""
        training_item = await self.get_item_from_request(req)
        return ait_http.create_response_from_item(training_item)

    async def on_post(self, req):
        """POST commands to an AI"""
        training_item = await self.get_item_from_request(req)

        req_url = req.url
        query = req_url.query
        try:
            command = query['command']
        except KeyError:
            ait_http.raise_bad_request('missing command in request')
        command_lower = command.lower()

        self.logger.info("{} command for AI {}/{}".format(
            command_lower, training_item.dev_id, training_item.ai_id))
        try:
            if command_lower == 'start':
                await self._start_training(training_item, query)
            elif command_lower == 'stop':
                await self._stop_training(training_item)
            else:
                ait_http.raise_bad_request('unknown command:' + command_lower)
        except ait.TrainingFailedError:
            ait_http.raise_bad_request_status("AI training state invalid",
                                              training_item.status,
                                              training_item.ai_id)

        return ait_http.create_response_from_item(training_item)

    async def _start_training(self, training_item, query):
        if training_item.is_training_active:
            self.logger.info(
                'Start command received, and already training, ignore it')
        else:
            if self.collection.is_training_slot_available():
                try:
                    max_training_mins = query['training_time_allowed']
                except KeyError:
                    max_training_mins = None
                self.logger.info(
                    'Start command received with max training time {} mins'.
                    format(max_training_mins))
                await training_item.start_training(max_training_mins)
            else:
                ait_http.raise_bad_request("No training slot available",
                                           web.HTTPTooManyRequests)

    async def _stop_training(self, training_item):
        if training_item.status.is_stopped:
            self.logger.info(
                'Stop command received, and already stopped training, '
                'ignore it')
        else:
            await training_item.stop_training()

    async def on_delete(self, req):
        """DELETE an AI"""
        training_item = await self.get_item_from_request(req)
        self.logger.info("Delete for AI {}/{}".format(training_item.dev_id,
                                                      training_item.ai_id))
        coro = self.collection.delete_item(training_item.dev_id,
                                           training_item.ai_id)
        asyncio.create_task(coro)
        resp = web.Response()
        return resp

    async def on_chat_v2(self, req):
        """ Post endpoint """
        if not req.can_read_body:
            self.logger.warning('No body found on chatv2')
            ait_http.raise_bad_request('No chat input provided')

        body = await req.json()

        # Ensure the correct json fields are present
        try:
            conversation = body['conversation']
            entities = body['entities']
        except AttributeError:
            # If we're missing these from the payload, its a bad request
            ait_http.raise_bad_request("Invalid chat payload", web.HTTPBadRequest)

        valid_data = True
        if type(entities) is str:
            # If no data is passed, the type will be str, but it should be empty
            # if so, ensure 'None' is passed down, anything else is an error
            if not entities:
                entities = None
            else:
                valid_data = False
        else:
            valid_data = self.check_valid_entity_data(entities)

        dev_id = req.match_info['dev_id']
        ai_id = req.match_info['ai_id']

        self.logger.info("on_chat_v2 called with payload '%s', ids '%s'/'%s'",
                         body,
                         dev_id,
                         ai_id,
                         extra={"dev_id": dev_id, "ai_id": ai_id})

        try:
            ai_hash = body['ai_hash']
        except KeyError:
            ai_hash = None

        if valid_data is True:
            resp = await self.do_chat(conversation, dev_id, ai_id, None, None, ai_hash, entities)
            resp = web.json_response(data=resp)
            return resp
        else:
            ait_http.raise_bad_request("Invalid entity data", web.HTTPBadRequest)

    def check_valid_entity_data(self, entities):
        valid_data = True
        try:
            # Validate entities is a dictionary of strings to list of strings
            for value, names in entities.items():
                if type(value) is not str:
                    valid_data = False
                if type(names) is not list:
                    valid_data = False
                for name in names:
                    if type(name) is not str:
                        valid_data = False
        except AttributeError:
            # If any of the above couldnt be processed, its not a payload we were expecting
            valid_data = False

        return valid_data

    async def on_chat(self, req):
        """GET on the chat endpoint"""
        req_url = req.url
        query = req_url.query

        try:
            ai_hash = query['ai_hash']
        except KeyError:
            # if no AI hash given, we ALWAYS reload training data in chat
            ai_hash = None

        # We have let some None values escape from Python, and API stores it.
        # None hash is treated the same as a missing hash.
        if ai_hash == "" or ai_hash == "None":
            ai_hash = None

        dev_id = req.match_info['dev_id']
        ai_id = req.match_info['ai_id']
        topic = query.get('topic', None)
        history = query.get('history', None)
        chat_input = query.get('q', None)
        if chat_input is None:
            ait_http.raise_bad_request('No chat input provided')

        self.logger.info("on_chat called with chat input '%s', ids '%s'/'%s'",
                         chat_input,
                         dev_id,
                         ai_id,
                         extra={"dev_id": dev_id, "ai_id": ai_id})

        resp = await self.do_chat(chat_input, dev_id, ai_id, topic, history, ai_hash, None)
        resp = web.json_response(data=resp)
        return resp

    async def do_chat(self, chat, dev_id, ai_id, topic, history, ai_hash, entities):
        item = self.collection.get_item(dev_id, ai_id)

        if item is None:
            # one attempt to load data
            ai_hash = None
            item = await self.collection.load_training_data_async(
                dev_id, ai_id)
            if item is None:
                raise web.HTTPNotFound()

        try:
            resp = await item.chat(chat, topic, history, ai_hash, entities)
        except ait_common.ChatStateError as exc:
            ait_http.raise_bad_request_status(exc.message, exc.state, exc.aiid)
        except ait_common.ChatAiHashError as exc:
            ait_http.raise_bad_request(str(exc))
        except ait_common.ChatOverloadedError as exc:
            error_data = exc.error_data
            error_data['message'] = exc.message
            data = {'status': '503', 'error': error_data}
            data_str = json.dumps(data)
            raise web.HTTPServiceUnavailable(text=data_str)
        return resp

    async def get_item_from_request(self, req):
        """Get item from request information, raises an HTTP not found error
        if not available"""
        dev_id = req.match_info['dev_id']
        ai_id = req.match_info['ai_id']
        training_item = self.collection.get_item(dev_id, ai_id)
        if training_item is None:
            training_item = await self.collection.load_training_data_async(
                dev_id, ai_id)
            if training_item is None:
                raise web.HTTPNotFound()
        return training_item
