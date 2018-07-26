import aiohttp
import dill
import string
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
        return [e for e in entities]

    async def tokenize(self, sample):
        tokens = await self.get_from_er_server("tokenize", {'q': sample})
        if not isinstance(tokens, list):
            raise EntityWrapperException(
                "Unexpected ER response - should be a list")
        return [token for token in tokens]

    def match_entities(self, test_q):
        max_matches = 0
        matched_labels = []
        test_match = test_q.lower()
        test_match = test_match.replace('"', '')
        test_match = test_match.replace("'s", "")
        test_match = test_match.replace("", "")
        d = test_match.maketrans('', '', string.punctuation)
        test_match = test_match.translate(d)
        test_match = test_match.split()
        self.logger.info("test_match: {}".format(test_match))
        for i, tr_ents in enumerate(self.train_entities):
            num_matches = 0
            self.logger.info("train sample ents: {}".format(tr_ents))
            for ent in tr_ents:
                if ent['category'] == 'sys.person':
                    tmp_ent = ent['value'].split()
                else:
                    tmp_ent = [ent['value']]
                for e in tmp_ent:
                    if e not in ['the'] and e in test_match:
                        num_matches += 1
            if num_matches > max_matches:
                matched_labels.append((i, self.train_labels[i]))
        return matched_labels

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
