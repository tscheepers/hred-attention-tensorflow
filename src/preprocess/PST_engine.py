import collections
import operator
import cPickle
import numpy as np


def kl_divergence(a, b, smoothing):
    # Approximate KL (i.e. smooth while computing)
    kl = 0.
    for x, v in a.items():
        kl += v * (np.log(v) - np.log(b.get(x, smoothing)))
    assert kl >= 0
    return kl


class PSTInfer(object):
    def __init__(self):
        self.query_to_id = {}
        self.id_to_query = []

    def _load_pickle(self, input_handle):
        self.tuple_dict = cPickle.load(input_handle)
        self.query_to_id = cPickle.load(input_handle)

    def load(self, input_path):
        print('Loading inference engine')

        input_handle = open(input_path, 'r')
        self._load_pickle(input_handle)
        input_handle.close()
        print('Preparing internal structures')

        # Transform the dict of tuples to a dict of dicts
        self.search_dict = collections.defaultdict(dict)
        for key, freq in self.tuple_dict.items():
            self.search_dict[key[:-1]][key[-1]] = freq

        self.tuple_dict.clear()
        self.id_to_query = [query_str for (query_str, query_id) in \
                            sorted(self.query_to_id.items(), key=operator.itemgetter(1))]

        print('Loaded inference engine')

    def _find(self, suffix, exact_match=False):
        _suffix = [self.query_to_id.get(x, -1) for x in suffix]

        # Back off to shorter suffixes,
        for i in range(len(_suffix)):
            key = tuple(_suffix[i:])
            if key in self.search_dict:
                return {'last_node': key, \
                        'is_found': i == 0 and len(_suffix) == len(suffix), \
                        'empty': False, \
                        'probs': self.search_dict[key]}
        # and if nothing is found
        return {'last_node': (0,), \
                'is_found': False, \
                'empty': True, \
                'probs': {}}

    def rerank(self, suffix, candidates, exact_match=False, no_normalize=False, fallback=False, dolog=False):
        probs = [self._find(suffix)]
        any_found = probs[0]['empty']
        found = probs[0]['is_found']

        if dolog:
            print("start: probs ", probs)
            print("any-found/found ", any_found, found)

        # for i in range(len(suffix)):
        #    probs.append(self._find(suffix[i:]))
        # any_found = sum([p['empty'] for p in probs])
        # Fallback to prefix matches
        if any_found and fallback:
            probs = []
            last_suffix = suffix[-1].split()
            while len(last_suffix) > 1:
                last_suffix = last_suffix[:-1]
                p = self._find(suffix[:-1] + [' '.join(last_suffix)])
                if not p['empty']:
                    probs = [p]
            if len(probs) == 0:
                print '!!!! Warning: should this be found instead ? ', suffix
        # If we don't find anything matching the suffix
        # we just return the original candidates
        if exact_match and not found:
            return [(candidate, 0) for candidate in candidates]
        ids_candidates = map(lambda x: self.query_to_id.get(x, -1), \
                             candidates)
        candidates_found = []
        candidates_not_found = []
        n_total_queries = len(self.id_to_query)

        for (id_candidate, candidate) in zip(ids_candidates, \
                                             candidates):
            # smoothed probability
            candidate_prob = 0
            for prob in probs:
                if id_candidate in prob['probs']:
                    # smooth and renormalize.
                    total_freq = 0
                    freq = 0
                    if no_normalize:
                        candidate_prob = prob['probs'][id_candidate]
                    else:
                        n_remaining_queries = (n_total_queries - len(prob['probs']))
                        assert n_remaining_queries >= 0
                        freq = prob['probs'][id_candidate]
                        total_freq = sum(prob['probs'].values())
                        candidate_prob = float(freq) / total_freq
                        candidate_prob = candidate_prob / (candidate_prob
                                                           + float(n_remaining_queries) / len(self.id_to_query))
                        candidate_prob = -np.log(candidate_prob)
                    if dolog:
                        print("n_total_queries %d/freq %.3f/total_freq %.3f/cand-prob %.3f" % (n_total_queries,
                                                                                               freq,
                                                                                               total_freq,
                                                                                               candidate_prob))
                    break

            if candidate_prob == 0 and not no_normalize:
                candidate_prob = -np.log(1.0 / n_total_queries)
                # candidate_prob = 1.0/n_total_queries
                if dolog:
                    print("Escape: %.3f" % candidate_prob)
            candidates_found.append((candidate, candidate_prob))
        return zip(*candidates_found)

    def suggest(self, suffix, N=100, exact_match=False):
        result = self._find(suffix)

        node = result['last_node']
        probs = result['probs']

        data = {'last_node_id': node[0],
                'last_node_query': self.id_to_query[node[0]],
                'found': result['is_found'],
                'suggestions': [],
                'scores': []}
        if node[0] == 0 or (exact_match and not data['found']):
            return data
        # Get top N
        id_sugg_probs = sorted(probs.items(), key=operator.itemgetter(1), reverse=True)[:N]
        string_sugg_probs = [(self.id_to_query[sugg_id], sugg_score) for sugg_id, sugg_score in id_sugg_probs]
        sugg, score = map(list, zip(*string_sugg_probs))
        data['suggestions'] = sugg
        data['scores'] = score

        return data


class PST(object):
    def __init__(self, D=4, q_dict=None):
        if q_dict is None:
            self.query_dict = {'_root_': 0}
        else:
            print("Parsing query dictionary!")
            self.query_dict = q_dict
        self.norm_dict = {}
        self.tuple_dict = {}
        self.normalized = False
        self.num_nodes = 1
        self.size = 0
        self.D = D

    def prune(self, epsilon=0.05):
        # Transform the dict of tuples to
        # a proper dict of dicts
        print('Started pruning with epsilon {}'.format(epsilon))
        search_dict = collections.defaultdict(lambda: {})
        for key, prob in self.tuple_dict.items():
            if not self.normalized:
                prob = float(prob) / self.norm_dict[key[:-1]]
            search_dict[key[:-1]][key[-1]] = prob
        self.tuple_dict.clear()
        print('Checking constistency')
        for key, prob in search_dict.items():
            assert np.abs(sum(prob.values()) - 1.0) < 1e-5
        self.normalized = True

        smoothing = 1.0 / len(self.query_dict)
        print('{} nodes / {} smoothing'.format(len(search_dict), smoothing))

        for num, (child_key, child_probs) in enumerate(search_dict.items()):
            if num % 100000 == 0:
                print('{} nodes explored'.format(num))
            # The parent of 1-length contexts is the root
            # thus we do not need to check here.
            if len(child_key) == 1:
                continue
            parent_key = child_key[1:]
            parent_probs = search_dict[parent_key]
            kl = kl_divergence(parent_probs, child_probs, smoothing)
            if kl <= epsilon:
                search_dict[child_key] = {}
        # Re-convert to tuple
        for num, (key, probs) in enumerate(search_dict.items()):
            for qid, qpr in probs.items():
                join_key = key + tuple([qid])
                assert len(join_key) >= 2
                assert join_key not in self.tuple_dict
                self.tuple_dict[join_key] = qpr
        # print('{} nodes - pruning done'.format(len(self.tuple_dict)))
        self.num_nodes = len(self.tuple_dict)

    def save(self, output_path, no_normalize=False):
        print('Saving PST to {} / {} nodes.'.format(output_path, len(self.tuple_dict)))
        # Save the normalized format
        # if not self.normalized and not no_normalize:
        #    print('Normalizing PST')
        #    for key, count in self.tuple_dict.iteritems():
        #        self.tuple_dict[key] = float(count) # /self.norm_dict.get(key[:-1])
        # self.norm_dict.clear()

        f = open(output_path, 'w')
        cPickle.dump(self.tuple_dict, f)
        cPickle.dump(self.query_dict, f)
        f.close()

    def add_session(self, session):
        def _update_prob(entry):
            key = entry[:-1]
            self.tuple_dict[entry] = self.tuple_dict.get(entry, 0) + 1
            self.norm_dict[key] = self.norm_dict.get(key, 0) + 1

        len_session = len(session)
        if len_session < 2:
            return
        for query in session:
            if query not in self.query_dict:
                self.query_dict[query] = len(self.query_dict)
        session = [self.query_dict[query] for query in session]
        # print("add session - session ", session, str(len_session))
        for x in range(len_session - 1):
            tgt_indx = len_session - x - 1
            for c in range(self.D):
                ctx_indx = tgt_indx - c - 1
                if ctx_indx < 0:
                    break

                entry = tuple(session[ctx_indx:tgt_indx + 1])
                # print("tuple entry ", entry)
                _update_prob(entry)

                self.num_nodes = len(self.tuple_dict)