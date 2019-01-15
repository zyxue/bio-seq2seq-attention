import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_attn(attns, src_seq, prd_seq, acc, time_step, outdir):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    matshow = ax.matshow(attns.numpy(), vmin=0, vmax=1)
    plt.colorbar(matshow, fraction=0.046, pad=0.04)

    ax.set_xticklabels([''] + list(src_seq), rotation=90)
    ax.set_yticklabels([''] + list(prd_seq))

    ax.set_xlabel('src_seq')
    ax.set_ylabel('prd_seq')

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_tick_params(labelsize=500 // len(src_seq),
                             labeltop=False, labelbottom=True)
    ax.yaxis.set_tick_params(labelsize=500 // len(src_seq))

    title = 'Step: {0:d}; Accuracy: {1:.3f}'.format(time_step, acc)
    ax.set_title(title, loc='left')

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    plt.savefig(os.path.join(outdir, '{0}.png'.format(time_step)))
    plt.close()
