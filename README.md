My implementation of seq2seq+attention models using
[PyTorch](https://pytorch.org/docs/stable/index.html), with
application for biological sequences in mind.

In contrast to a general seq2seq problem (e.g. translation), the input
and output bio sequences are already aligned, e.g. gene prediction,
protein secondary structure prediction, etc, and the vocabulary sizes
tend to much smaller.


#### Development

```
python setup.py develop
```
