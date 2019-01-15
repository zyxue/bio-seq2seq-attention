class Language(object):
    def __init__(self, name, vocab, beg_token='^', end_token='$', unk_token='*'):
        """
        :param name: give the language a name
        :param vocab: vocabulary, e.g. set{'A', 'T', 'C', 'G'}
        :param beg_token: token that indicates the beginning of a sequence
        :param end_token: token that indicates the end of a sequence
        :param unk_token: token that indicates the unknown/missing parts of a sequence
        """
        self.name = name
        # convention: vocabulary doesn't include beg/end/unk tokens
        self.vocab = set(vocab)
        self.beg_token = beg_token
        self.end_token = end_token
        self.unk_token = unk_token

        self.init_indices()

        self.num_tokens = len(self.token2index)

    def init_indices(self):
        self.token2index = self.gen_t2i_dict()
        self.index2token = self.gen_i2t_dict()

    def gen_t2i_dict(self):
        dd = {}
        sorted_tokens = [self.beg_token,
                         self.end_token,
                         self.unk_token] + list(sorted(self.vocab))
        for k, i in enumerate(sorted_tokens):
            dd[i] = k
        return dd

    def gen_i2t_dict(self):
        return {j: i for (i, j) in self.token2index.items()}

    def __repr__(self):
        return f'Language("{self.name}", "{self.vocab}")'

    def __str__(self):
        return f'Language: {self.name}, {len(self.vocab)} tokens'
