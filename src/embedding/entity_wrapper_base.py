import aiohttp
import logging

"""
This is a wrapper which contains the client to contact the ER RestAPI
which does the entity recognition as well as the sentence tokenization.
"""


class EntityWrapperException(Exception):
    pass


class EntityWrapper(object):
    def __init__(self, service_url, client_session: aiohttp.ClientSession):
        self.service_url = service_url
        self.logger = logging.getLogger('entity_matcher')
        self.client_session = client_session

    async def get_from_er_server(self, relative_url, params=None):
        try:
            async with self.client_session.get(
                    self.service_url + "/" + relative_url,
                    params=params) as resp:
                status = resp.status
                if status != 200:
                    raise EntityWrapperException(
                        "ER call to {} failed with status {}".format(
                            relative_url, status))
                response = await resp.json()

            if response is None:
                raise EntityWrapperException("Response was none")

            return response
        except (aiohttp.client_exceptions.ClientConnectorError,
                aiohttp.client_exceptions.ContentTypeError) as exc:
            raise EntityWrapperException("aiohttp error", exc)

    async def extract_entities(self, sample):
        entities = await self.get_from_er_server("ner", {'q': sample})
        if not isinstance(entities, list):
            raise EntityWrapperException(
                "Unexpected ER response - should be a list")
        for e in entities:
            if "'s" in e['value']:
                e['value'] = e['value'].replace("'s", "")
            e['value'] = e['value'].lower()
            if e['value'].startswith('the'):
                e['value'] = e['value'].replace('the', '')
        self.logger.debug("entities for '{}': {}".format(sample, entities))
        return entities

    async def tokenize(self, sample, filter_ents='True', sw_size='small'):
        tokens = await self.get_from_er_server("tokenize", {
            'q': sample,
            'filter_ents': filter_ents,
            'sw_size': sw_size
        })
        if not isinstance(tokens, list):
            raise EntityWrapperException(
                "Unexpected ER response - should be a list")
        return [token for token in tokens]
