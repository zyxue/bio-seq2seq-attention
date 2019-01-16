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
        self.vocab = sorted(set(vocab))
        self.beg_token = beg_token
        self.end_token = end_token
        self.unk_token = unk_token

        # tokens include every character while vocab doesn't
        self.tokens = [beg_token, end_token, unk_token] + self.vocab

        self.init_indices()

        self.num_tokens = len(self.token2index)

    def init_indices(self):
        self.token2index = self.gen_t2i_dict()
        self.index2token = self.gen_i2t_dict()

    def gen_t2i_dict(self):
        dd = {}
        for k, i in enumerate(self.tokens):
            dd[i] = k
        return dd

    def gen_i2t_dict(self):
        return {j: i for (i, j) in self.token2index.items()}

    def to_dict(self):
        return dict(
            name=self.name,
            vocab=self.vocab,
            beg_token=self.beg_token,
            end_token=self.end_token,
            unk_token=self.unk_token
        )

    def __repr__(self):
        return f'Language("{self.name}", "{self.vocab}")'

    def __str__(self):
        return f'Language: {self.name}, {len(self.vocab)} tokens'

    def seq2indices(self, seq):
        return [self.token2index[i] for i in seq]
