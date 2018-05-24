import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


class Lang(object):
    def __init__(self, name, vocab_file):
        self.name = name
        self.vocab_file = vocab_file

        self.word2index = {}
        with open(vocab_file) as inf:
            for k, i in enumerate(inf):
                i = i.strip()
                if not i:       # empty line
                    continue
                self.word2index[i] = k

        self.index2word = {j: i for (i, j) in self.word2index.items()}

        self.n_words = len(self.word2index)

    def __repr__(self):
        return 'Lang("{0}", "{1}")'.format(self.name, self.vocab_file)

    def __str__(self):
        return 'Lang: {0}, {1} words'.format(self.name, self.n_words)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# # ### Evaluation

# # In[22]:


# def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
#     with torch.no_grad():
#         input_tensor = tensor_from_sentence(input_lang, sentence)
#         input_length = input_tensor.size()[0]
#         encoder_hidden = encoder.initHidden()

#         encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

#         for ei in range(input_length):
#             encoder_output, encoder_hidden = encoder(input_tensor[ei],
#                                                      encoder_hidden)
#             encoder_outputs[ei] += encoder_output[0, 0]

#         decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

#         decoder_hidden = encoder_hidden

#         decoded_words = []
#         decoder_attentions = torch.zeros(max_length, max_length)

#         for di in range(max_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(
#                 decoder_input, decoder_hidden, encoder_outputs)
#             decoder_attentions[di] = decoder_attention.data
#             topv, topi = decoder_output.data.topk(1)
#             if topi.item() == EOS_token:
#                 decoded_words.append('<EOS>')
#                 break
#             else:
#                 decoded_words.append(output_lang.index2word[topi.item()])

#             decoder_input = topi.squeeze().detach()

#         return decoded_words, decoder_attentions[:di + 1]


# # In[23]:


# def evaluateRandomly(encoder, decoder, n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('>', pair[0])
#         print('=', pair[1])
#         output_words, attentions = evaluate(encoder, decoder, pair[0])
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')

