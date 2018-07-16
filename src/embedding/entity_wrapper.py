import aiohttp
import dill
import logging
from pathlib import Path


class EntityWrapperException(Exception):
    pass


class EntityWrapper:

    def __init__(self, service_url, client_session: aiohttp.ClientSession):
        self.service_url = service_url
        self.logger = logging.getLogger('entity_matcher')
        self.client_session = client_session
        self.train_labels = None
        self.train_entities = None

    async def get_from_er_server(self, relative_url, params=None):
        try:
            async with self.client_session.get(
                    self.service_url + "/" + relative_url, params=params) as resp:
                status = resp.status
                if status != 200:
                    raise EntityWrapperException(
                        "ER call to {} failed with status {}".format(relative_url, status))
                response = await resp.json()

            if response is None:
                raise EntityWrapperException("Response was none")

            return response
        except (aiohttp.client_exceptions.ClientConnectorError
                | aiohttp.client_exceptions.ContentTypeError) as exc:
            raise EntityWrapperException("aiohttp error", exc)

    async def extract_entities(self, sample):
        entities = await self.get_from_er_server("ner", {'q': sample})
        if not isinstance(entities, list):
            raise EntityWrapperException("Unexpected ER response - should be a list")
        return [e["value"] for e in entities]

    async def tokenize(self, sample):
        tokens = await self.get_from_er_server("tokenize", {'q': sample})
        if not isinstance(tokens, list):
            raise EntityWrapperException("Unexpected ER response - should be a list")
        return [token for token in tokens]

    def match_entities(self, test_q):
        max_matches = 0
        matched_label = None
        for i, tr_ents in enumerate(self.train_entities):
            num_matches = sum([(e in test_q or e.lower() in test_q) for e in tr_ents])
            if num_matches > max_matches:
                matched_label = self.train_labels[i]
        return matched_label

    def save_data(self, file_path: Path, ents, train_labels):
        if not isinstance(ents, list):
            self.logger.error('data to save must be list')
            raise EntityWrapperException('data to save must be list')
        with file_path.open('wb') as f:
            dill.dump([ents, train_labels], f)

    def load_data(self, file_path: Path):
        with file_path.open('rb') as f:
            d = dill.load(f)
        self.train_entities = d[0]
        self.train_labels = d[1]
