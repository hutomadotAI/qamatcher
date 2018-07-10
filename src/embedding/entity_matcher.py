import dill
import logging


class EntityMatcher(object):

    def __init__(self, spacy):
        self.train_labels = None
        self.logger = logging.getLogger('entity_matcher')
        self.spacy = spacy

    def extract_entities(self, sample):
        ents = self.spacy.parser(sample).ents
        return [e.text for e in ents]

    def match_entities(self, train_ents, test_q):
        max_matches = 0
        matched_label = None
        for i, tr_ents in enumerate(train_ents):
            num_matches = sum([(e in test_q or e.lower() in test_q) for e in tr_ents])
            if num_matches > max_matches:
                matched_label = self.train_labels[i]
        return matched_label

    def save_data(self, file_path, ents, train_labels):
        assert isinstance(ents, list), 'data to save must be list'
        with open(file_path, 'wb') as f:
            dill.dump([ents, train_labels], f)

    def load_data(self, file_path):
        with open(file_path, 'rb') as f:
            d = dill.load(f)
        ents = d[0]
        self.train_labels = d[1]
        return ents