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

    def __prepro_question(self, test_q):
        test_match = test_q.lower()
        test_match = test_match.replace('"', '')
        test_match = test_match.replace("'s", "")
        test_match = test_match.replace("", "")
        d = test_match.maketrans('', '', '!"\'(),./:;<=>?[\\]`{|}')
        test_match = test_match.translate(d)
        test_match = test_match.split()
        return test_match

    def match_entities(self, test_q, ents_msg, subset_idxs=None):
        test_match = self.__prepro_question(test_q)
        self.logger.debug("test_match: {}".format(test_match))

        # subset if already pre-selected using different algo
        sub_idxs = subset_idxs if subset_idxs is not None else list(
            range(len(self.train_entities_a)))
        ents_q_a = [(i, self.train_entities_q[i], self.train_entities_a[i])
                    for i in sub_idxs]

        # search for interrogative words matching entitites
        interrog_matches = [(i, ents_q,
                             self.interrogative_match(
                                 test_match, (ents_msg, ents_q, ents_a)))
                            for i, ents_q, ents_a in ents_q_a]
        _, _, cnt = zip(*interrog_matches)
        self.logger.debug("interrog count: {}".format(cnt))
        max_cnt = max(cnt)
        interrog_matches = [(m[0], m[1]) for m in interrog_matches
                            if m[2] == max_cnt and max_cnt > 0]
        self.logger.debug("interrog matches: {}".format(interrog_matches))
        if len(interrog_matches) > 0:
            train_ents = interrog_matches
        else:
            train_ents = [(i, ents_q) for i, ents_q, ents_a in ents_q_a]

        # search for entity matches between train and test q's
        matched_labels = self.find_matches(train_ents, test_match)

        if len(matched_labels) > 0:
            self.logger.debug("entity matches: {} ({} max matches)".format(
              matched_labels, max_cnt))
            return matched_labels
        elif len(interrog_matches) > 0:
            self.logger.debug("interrog matches: {}".format(
              [(i, self.train_labels[i]) for i, _ in interrog_matches]))
            return [(i, self.train_labels[i]) for i, _ in interrog_matches]
        else:
            self.logger.debug("no entity matches")
            return []

    def find_matches(self, train_ents, test_match):
        max_matches = 0
        matched_labels = []
        for i, tr_ents in train_ents:
            num_matches = 0
            self.logger.debug("train sample ents: {}".format(tr_ents))
            num_matches += sum(1 if e not in ['the'] and e in test_match else 0
                               for ent in tr_ents for e in self.split_entities(ent))
            if num_matches > max_matches:
                max_matches = num_matches
                matched_labels = [(i, self.train_labels[i])]
            elif num_matches == max_matches and max_matches > 0:
                matched_labels.append((i, self.train_labels[i]))
        return matched_labels

    def split_entities(self, ent):
        if ent['category'] in [
                'sys.person', 'sys.group', 'sys.organization'
        ]:
            tmp_ent = ent['value'].split()
        else:
            tmp_ent = [ent['value']]
        return tmp_ent

    def interrogative_match(self, test_match, ents):
        match = 0
        match += int(self.check_who_questions(test_match, ents))
        match += int(self.check_for_person(test_match, ents))
        match += int(self.check_for_custom_entity(test_match, ents))
        match += int(self.check_who_questions_inv(test_match, ents))
        # match += int(self.check_what_questions(test_match, ents))
        return match

    def check_who_questions(self, test_match, ents):
        ents_msg, ents_q, ents_a = ents
        match = False
        if 'who' in test_match and not any(
                [e['category'] == 'sys.person' for e in ents_msg]):
            match = any(
                [ent_a['category'] == 'sys.person' for ent_a in ents_a])
        return match

    def check_who_questions_inv(self, test_match, ents):
        ents_msg, ents_q, ents_a = ents
        match = False
        if 'who' in test_match:
            match = any([
                e in test_match for ent in ents_q
                for e in ent['value'].split()
                if ent['category'] == 'sys.person'
            ])
        return match

    def check_what_questions(self, test_match, ents):
        ents_msg, ents_q, ents_a = ents
        match = False
        orgs = [
            e['value'].lower() for e in ents_msg
            if e['category'] == ['sys.group', 'sys.organization']
        ]
        if 'what' in test_match and len(orgs) > 0:
            match = any([ent_q['value'] in orgs for ent_q in ents_q])
        return match

    def check_for_person(self, test_match, ents):
        ents_msg, ents_q, ents_a = ents
        person = [
            e['value'].lower().split() for e in ents_msg
            if e['category'] == 'sys.person'
        ]
        person = [e for p in person for e in p]
        if len(person) > 0:
            return any(
                [p in e['value'].lower() for e in ents_q for p in person])
        else:
            return False

    def check_for_custom_entity(self, test_match, ents):
        ents_msg, ents_q, ents_a = ents
        cust_ents = [
            e['value'] for e in ents_msg if e['category'].startswith("@")
        ]
        if len(cust_ents) > 0:
            return any([e['value'] in cust_ents for e in ents_a])
        else:
            return False

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
