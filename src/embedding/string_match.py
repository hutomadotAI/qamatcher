import logging
import dill


class StringMatchException(Exception):
    pass


class StringMatch:
    def __init__(self, entity_wrapper):
        self.logger = logging.getLogger('string_match')
        self.train_data = None
        self.tok_train = []
        self.entity_wrapper = entity_wrapper
        # self.stopword_size = 'small'
        # self.filter_entities = 'False'

    def load_train_data(self, file_path):
        with file_path.open('rb') as f:
            self.train_data = dill.load(f)

    def save_train_data(self, data, file_name):
        if not isinstance(data, list):
            self.logger.error('data to save must be list')
            raise TypeError('data to save must be list')
        self.logger.info('saving training file to {}'.format(file_name))
        with open(file_name, 'wb') as f:
            dill.dump(data, f)

    async def tokenize_train_data(self):
        for q in self.train_data:
            # tok = await self.entity_wrapper.tokenize(
            #     q[0],
            #     # filter_ents=self.filter_entities,
            #     # sw_size=self.stopword_size
            # )
            tok = q[0].lower().split()
            self.tok_train.append(tok)

    async def get_string_match(self, q, subset_idx=None,
                               all_larger_zero=False):
        self.logger.info("searching for word matches")
        tok_train = self.tok_train if subset_idx is None else [
            self.tok_train[i] for i in subset_idx
        ]
        train_data = self.train_data if subset_idx is None else [
            self.train_data[i] for i in subset_idx
        ]
        idx = subset_idx if subset_idx is not None else range(len(train_data))
        tok_q = q.lower().split()  # await self.entity_wrapper.tokenize(
            # q)  # , filter_ents=self.filter_entities, sw_size=self.stopword_size)

        # search for intent-like entities first
        if "@" in q:
            match_probas = [
                1.0 if "@" in t[0] else 0.0 for t in self.train_data
            ]
        # otherwise do string match
        else:
            match_probas = [
                self.__jaccard_similarity(tok_q, t)
                if '@' not in ' '.join(t) else 0.0 for t in tok_train
            ]

        self.logger.info("match_probas: {}".format(match_probas))
        max_proba = max(match_probas)
        if all_larger_zero:
            def f(a, b): return a > 0.
        else:
            def f(a, b): return a == b
        preds = [(idx[i], train_data[i][1]) for i, p in enumerate(match_probas)
                 if f(p, max_proba)]
        self.logger.info("string_match: {} - {}".format(max_proba, preds))
        return max_proba, preds

    def __jaccard_similarity(self, list1, list2):
        a = set(list1)
        b = set(list2)
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
