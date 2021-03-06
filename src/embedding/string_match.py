import logging
import dill


class StringMatchException(Exception):
    pass


class StringMatch:
    def __init__(self, entity_wrapper):
        self.logger = logging.getLogger('string_match')
        self.train_data = None
        self.tok_train = None
        self.tok_train_no_sw = None
        self.cust_ents_train = None
        self.entity_wrapper = entity_wrapper
        self.stopword_size = 'large'
        self.filter_entities = 'False'

    def load_train_data(self, file_path):
        with file_path.open('rb') as f:
            tmp = dill.load(f)
        self.train_data = tmp[0]
        self.tok_train = tmp[1]
        self.cust_ents_train = tmp[2]
        self.tok_train_no_sw = tmp[3]

    def save_train_data(self, data, file_name):
        if not isinstance(data, list):
            self.logger.error('data to save must be list')
            raise TypeError('data to save must be list')
        self.logger.info('saving training file to {}'.format(file_name))
        with open(file_name, 'wb') as f:
            dill.dump(data, f)

    async def get_string_match(self, q, subset_idx=None,
                               all_larger_zero=False, entities=None):
        self.logger.info("searching for word matches")
        tok_train, tok_train_no_sw, train_data, idx = self.subset_data(subset_idx)
        tok_q = await self.entity_wrapper.tokenize(
            q, filter_ents=self.filter_entities, sw_size=self.stopword_size)
        tok_q_no_sw = await self.entity_wrapper.tokenize(
            q, sw_size='small', filter_ents='False')

        match_probas = []
        for t_cust_ents, t_tok, t_tok_no_sw in \
                zip(self.cust_ents_train, tok_train, tok_train_no_sw):
            self.logger.debug("t_cust_ents: {}".format(t_cust_ents))
            # if perfect match between train and query use that
            score = self.__jaccard_similarity(tok_q_no_sw, t_tok_no_sw)
            self.logger.debug("tok_q_no_sw: {} t_tok_no_sw: {}".format(tok_q_no_sw, t_tok_no_sw))
            if score > 0.99:
                self.logger.debug("exact match: {} == {}".format(tok_q_no_sw, t_tok_no_sw))
                # give score of 2 to make absolutely sure that this one is accepted
                match_probas.append(2.)
                continue
            cust_ent_score = await self.match_custom_entities(
                entities, t_cust_ents, t_tok, q)
            if cust_ent_score is not None:
                match_probas.append(cust_ent_score)
            elif tok_q[0] != 'UNK' and t_tok[0] != 'UNK':
                score = self.__jaccard_similarity(tok_q, t_tok)
                self.logger.debug("raw score: %f", score)
                match_probas.append(score if len(t_cust_ents) == 0 else
                                    score - 0.2 * score)
            else:
                match_probas.append(0.0)

        self.logger.info("match_probas: {}".format(match_probas))
        max_proba = max(match_probas)
        if all_larger_zero:
            def f(a, b): return a > 0.
        else:
            def f(a, b): return a == b
        preds = [(idx[i], train_data[i][1]) for i, p in enumerate(match_probas)
                 if f(p, max_proba)]

        # if all found matches have the same answer just pick first one
        if len(set([p[1] for p in preds])) == 1:
            preds = [preds[0]]
        self.logger.info("string_match: {} - {}".format(max_proba, preds))
        return max_proba, preds

    async def match_custom_entities(self, entities, t_cust_ents, t_tok, q):
        if entities:
            matching_ents = {k.lower(): e for k, v in entities.items()
                             for e in v if e in t_cust_ents}
        else:
            matching_ents = {}
        self.logger.info("matching_ents: {}".format(matching_ents))
        if len(matching_ents) > 0:
            subst_query = q
            for k, e in matching_ents.items():
                w = subst_query.lower().find(k)
                if w < 0:
                    self.logger.warning("matched entities not found")
                    continue
                subst_query = subst_query[:w] + '@{' + e + '}@' + subst_query[w+len(k):]
            tok_subst_q = await self.entity_wrapper.tokenize(
                subst_query, filter_ents=self.filter_entities, sw_size=self.stopword_size)
            self.logger.debug("subst_query: {}".format(subst_query))
            self.logger.debug("tok_subst_query: {}".format(tok_subst_q))
            self.logger.debug("t_tok: {}".format(t_tok))
            score = self.__jaccard_similarity(tok_subst_q, t_tok)
            self.logger.debug("raw score: %f", score)
            return (score + min(0.5 * (1 - score), 0.3) *
                    len(matching_ents) / len(t_cust_ents))
        else:
            return None

    def subset_data(self, subset_idx):
        tok_train = self.tok_train if subset_idx is None else [
            self.tok_train[i] for i in subset_idx
        ]
        tok_train_no_sw = self.tok_train_no_sw if subset_idx is None else [
            self.tok_train_no_sw[i] for i in subset_idx
        ]
        train_data = self.train_data if subset_idx is None else [
            self.train_data[i] for i in subset_idx
        ]
        idx = subset_idx if subset_idx is not None else range(len(train_data))
        return tok_train, tok_train_no_sw, train_data, idx

    def __jaccard_similarity(self, list1, list2):
        a = set(list1)
        b = set(list2)
        c = a.intersection(b)
        self.logger.debug("a: {} b: {} c: {}".format(a, b, c))
        return float(len(c)) / (len(a) + len(b) - len(c))
