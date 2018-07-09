import dill
import logging
import numpy as np

from spacy_wrapper import SpacyWrapper


class EntityMatcher(object):

    def __init__(self):
        self.train_labels = None
        self.tokenizer = SpacyWrapper()
        self.logger = logging.getLogger('entity_matcher')

    def extract_entities(self, sample):
        ents = self.tokenizer.extract_entities(sample)
        return [e.text for e in ents]

    def match_entities(self, train_ents, test_ents):
        matches = np.array([sum([e in tr_ents for e in test_ents]) for tr_ents in train_ents])
        if max(matches):
            matched_label = self.train_labels[np.argmax(matches)]
        else:
            matched_label = None
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