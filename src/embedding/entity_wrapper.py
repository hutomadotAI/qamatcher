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
        self.train_entities_q = None
        self.train_entities_a = None

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
        self.logger.info("entities for '{}': {}".format(sample, entities))
        return entities

    async def tokenize(self, sample, filter_ents='True', sw_size='small'):
        tokens = await self.get_from_er_server("tokenize", {'q': sample,
                                                            'filter_ents': filter_ents,
                                                            'sw_size': sw_size})
        if not isinstance(tokens, list):
            raise EntityWrapperException(
                "Unexpected ER response - should be a list")
        return [token for token in tokens]

    def __prepro_question(self, test_q):
        test_match = test_q.lower()
        test_match = test_match.replace('"', '')
        test_match = test_match.replace("'s", "")
        test_match = test_match.replace("", "")
        d = test_match.maketrans('', '', '!"\'(),./:;<=>?[\\]`{|}')
        test_match = test_match.translate(d)
        test_match = test_match.split()
        return test_match

    def match_entities(self, test_q, subset_idxs=None):
        test_match = self.__prepro_question(test_q)
        max_matches = 0
        matched_labels = []
        self.logger.info("test_match: {}".format(test_match))

        # subset if already pre-selected using different algo
        sub_idxs = subset_idxs if subset_idxs is not None else list(range(len(self.train_entities_a)))
        ents_q_a = [(i, self.train_entities_q[i], self.train_entities_a[i]) for i in sub_idxs]

        # search for interrogative words matching entitites
        interrog_matches = [(i, ents_q) for i, ents_q, ents_a in ents_q_a
                              if self.interrogative_match(test_match, ents_q, ents_a)]
        if len(interrog_matches) > 0:
            train_ents = interrog_matches
        else:
            train_ents = [(i, ents_q) for i, ents_q, ents_a in ents_q_a]

        # search for entity matches between q&a
        for i, tr_ents in train_ents:
            num_matches = 0
            # self.logger.info("train sample ents: {}".format(tr_ents))
            for ent in tr_ents:
                if ent['category'] in ['sys.person', 'sys.group', 'sys.organization']:
                    tmp_ent = ent['value'].split()
                else:
                    tmp_ent = [ent['value']]
                for e in tmp_ent:
                    if e not in ['the'] and e in test_match:
                        num_matches += 1
            if num_matches > max_matches:
                max_matches = num_matches
                matched_labels = [(i, self.train_labels[i])]
            elif num_matches == max_matches and max_matches > 0:
                matched_labels.append((i, self.train_labels[i]))

        self.logger.info("entity matches: {} ({} matches)".format(matched_labels, max_matches))

        if len(matched_labels) > 0:
            return matched_labels
        elif len(interrog_matches) > 0:
            return [(i, self.train_labels[i]) for i, _ in interrog_matches]
        else:
            return []

    def interrogative_match(self, test_match, ents_q, ents_a):
        match = False
        if 'who' in test_match:
            match = any([ent_a['category'] == 'sys.person' and not
                         any([ent_q['category'] != 'sys.person' for ent_q in ents_q])
                         for ent_a in ents_a])
        # elif 'what' in test_match:
        #     match = any([ent['category'] in ['sys.group', 'sys.organization'] for ent in train_ents])
        return match

    def interrogative_word_match(self, test_q, subset_idxs=None):
        train_ents = [self.train_entities_a[i] for i in subset_idxs] if subset_idxs else self.train_entities_a
        train_labels = [self.train_labels[i] for i in subset_idxs] if subset_idxs else self.train_labels
        matches = []
        if 'who' in test_q.lower():
            matches = [any([True if e['category'] == 'sys.person' else False
                            for e in ents]) for ents in train_ents]
        elif 'what' in test_q.lower():
            matches = [any([True if e['category'] in ['sys.group', 'sys.organization'] else False
                            for e in ents]) for ents in train_ents]

        if sum(matches) > 0:
            pred = [(i, train_labels[i]) for i, b in enumerate(matches) if b is True]
        else:
            pred = []
        return pred

    def save_data(self, file_path: Path, q_ents, a_ents, train_labels):
        if not isinstance(q_ents, list) or not isinstance(a_ents, list):
            self.logger.error('data to save must be list')
            raise EntityWrapperException('data to save must be list')
        with file_path.open('wb') as f:
            dill.dump([q_ents, a_ents, train_labels], f)

    def load_data(self, file_path: Path):
        with file_path.open('rb') as f:
            d = dill.load(f)
        self.train_entities_q = d[0]
        self.train_entities_a = d[1]
        self.train_labels = d[2]
