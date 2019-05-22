import abc
import asyncio
import json
import logging
import random
import typing

import aiohttp

import ai_training


def _get_logger():
    logger = logging.getLogger('hu.api_register')
    return logger


class ApiAiStatus:
    """Ai Status for a single AI"""

    def __init__(self,
                 training_status=ai_training.AiTrainingState.ai_undefined,
                 ai_id=None,
                 ai_hash=None):
        self.training_status = training_status
        self.ai_id = ai_id
        self.ai_hash = ai_hash


class AiStatusProviderABC(abc.ABC):
    """Provider for information about AI status"""

    @abc.abstractmethod
    async def get_ai_statuses_for_api(self) -> typing.List[ApiAiStatus]:
        """Get a mapping of ai status"""


class Register:
    def __init__(self, training_enabled: bool, chat_enabled: bool,
                 ai_engine_type: str, this_service_url: str, language: str,
                 version: str, provider: AiStatusProviderABC):
        self.api_endpoint = None
        self.training_enabled = training_enabled
        self.chat_enabled = chat_enabled
        self.ai_engine_type = ai_engine_type
        self.this_service_url = this_service_url
        self.language = language
        self.version = version
        self.provider = provider
        self.logger = _get_logger()
        self._shutdown_flag = False
        self.session_id = None

    async def registration_loop(self, http_session):
        success = False
        if self.api_endpoint is None:
            # API not set, return
            self.logger.warning("Can't register with API, endpoint not set")
            return

        log_registration_json = True
        while not success:
            if self._shutdown_flag:
                return
            success = await self.register(http_session, log_registration_json)
            log_registration_json = False
            if not success:
                # sleep between 1 and 2 seconds
                sleep_time = random.uniform(1.0, 2.0)
                self.logger.info('Sleeping for {:.2f}s'.format(sleep_time))
                await asyncio.sleep(sleep_time)

    def set_shutdown(self):
        self._shutdown_flag = True

    async def register(self, http_session, log_registration_json=False):
        # clear existing session ID
        self.session_id = None
        ai_engine_type = self.ai_engine_type
        this_service_url = self.this_service_url
        url = '{}/aiservices/register'.format(self.api_endpoint)

        training_enabled = self.training_enabled
        chat_enabled = self.chat_enabled
        ai_statuses = await self.provider.get_ai_statuses_for_api()
        ai_data = [{
            'ai_id': ai.ai_id,
            'training_status': ai.training_status,
            'ai_hash': ai.ai_hash
        } for ai in ai_statuses]
        data = {
            'server_type': ai_engine_type,
            'server_url': this_service_url,
            'training_capacity': 1 if training_enabled else 0,
            'chat_capacity': 1 if chat_enabled else 0,
            'language': self.language,
            'version': self.version,
            'ai_list': ai_data
        }
        json_data = json.dumps(data)

        register_success = False
        self.logger.info("Registering to {}".format(url))
        headers = {'content-type': 'application/json'}
        if log_registration_json:
            self.logger.info("Registration JSON='{}'".format(json_data))
        try:
            async with http_session.post(
                    url, data=json_data, headers=headers) as response:
                status = response.status
                if status == 200:
                    json_response = await response.json()
                    self.session_id = json_response['server_session_id']
                    register_success = True
                    self.logger.info(
                        "Register to {} successful: session is '{}'".format(
                            url, self.session_id))
                elif status == 400:
                    text_response = await response.text()
                    self.logger.error(
                        "Register to {} failed with status 400: {}".format(
                            url, text_response))
                else:
                    self.logger.error(
                        "Register to {} failed with status {}".format(
                            url, status))
        except (aiohttp.ClientOSError, aiohttp.ClientResponseError) as exc:
            self.logger.error("Update to {} failed with exception: {}".format(
                url, type(exc)))
        except Exception:
            self.logger.error(
                "Update to {} failed with unexpected exception".format(url),
                exc_info=True)
        return register_success
