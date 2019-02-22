# Update 2019-02-20

Started trying to implement a seq-to-seq model for labelling each base of a bio
sequence (e.g. transcript) see tag
[seq2seq](https://github.com/zyxue/bio-seq2seq-attention/tree/seq2seq), it
didn't work quite well, so currently trying to implement RNN+MLP architecture.

-----


My implementation of seq2seq+attention models using
[PyTorch](https://pytorch.org/docs/stable/index.html), with
application for biological sequences in mind.

In contrast to a general seq2seq problem (e.g. translation), the input
and output bio sequences are already aligned, e.g. gene prediction,
protein secondary structure prediction, etc, and the vocabulary sizes
tend to much smaller.


#### size variable naming convention

```
L: seq_len
B: batch_size
E: embedding size
H: hidden size
D: num_directions
Y: num_hidden_layers
C: num_tokens/classes
```

#### Development

```
python setup.py develop
```
