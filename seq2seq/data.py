import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


class SeqData(Dataset):
    def __init__(self, lang0, lang1, csv_file):
        self.lang0 = lang0
        self.lang1 = lang1

        res = []
        with open(csv_file, 'rt') as inf:
            for k, line in tqdm(enumerate(inf)):
                if k % 2 == 0:
                    seq, _ = line.strip().split()
                    _r = [seq]
                else:
                    lab, _ = line.strip().split()
                    _r.append(lab)
                    res.append(_r)
        self.data_frame = pd.DataFrame(res, columns=['seq', 'lab'])

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        seq, lab = self.data_frame.iloc[idx].values.tolist()
        seq_idx = self.lang0.seq2indices(seq)
        lab_idx = self.lang1.seq2indices(lab)
        return [seq_idx, lab_idx]



def pad_seqs(seqs):
    max_len = max(len(i) for i in seqs)

    # 2 corresponds to the unk_token.
    # TODO: replace 2 with something more sensible, or rewrite data.py with
    # some pytorch-provided class
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
