import torch


def pad_seqs(seqs):
    max_len = max(len(i) for i in seqs)

    # 2 corresponds to the unk_token. TODO: replace 2 with something more sensible
    seqs = [i + [2] * (max_len - len(i)) for i in seqs]
    return seqs


def convert_to_tensor(seqs, device):
    # should be of shape (seq_len, batch, 1) based on pytorch convention:
    # https://pytorch.org/docs/stable/nn.html#torch.nn.GRU
    return torch.tensor(seqs, device=device).transpose(1, 0)


def prep_training_data(lang0, lang1, data_file, batch_size, device):
    """
    prepare training data in tensors, returns an infinite generator

    :param seq_pairs: a list of (seq0, seq1, length) tuples
    """
    # assuming lines in data_file are already shuffled
    seq0s, seq1s, seq_lens, counter = [], [], [], 0
    while True:
        with open(data_file, 'rt') as inf:
            for line in inf:
                seq0, seq1 = line.strip().split()
                assert len(seq0) == len(seq1)

                seq0s.append(lang0.seq2indices(seq0))
                seq1s.append(lang1.seq2indices(seq1))
                seq_lens.append(len(seq0))
                counter += 1

                if counter == batch_size:
                    seq0s = convert_to_tensor(pad_seqs(seq0s), device)
                    seq1s = convert_to_tensor(pad_seqs(seq1s), device)
                    seq_lens = torch.tensor(seq_lens, device=device)
                    yield [seq0s, seq1s, seq_lens]

                    # reset
                    seq0s, seq1s, seq_lens, counter = [], [], [], 0
