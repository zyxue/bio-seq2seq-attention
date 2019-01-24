def encode(encoder, data_batch):
    """
    run data through encoder and return the encoded tensors

    data_batch is a tuple of (seq0s, seq1s, seq_lens).
    seq0s and seq1s should be in indices
    """
    seq0s, _, _ = data_batch
    # seq0s.shape: L x B
    _, B = seq0s.shape

    hid = encoder.init_hidden(B, device=seq0s.device)
    out, hid = encoder(seq0s, hid)

    return out, hid
