import os


idx = 0
lrs = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
lrs = [0.005]
for lr, num_iters in zip(lrs, [1000 for _ in lrs]):
    for E in [4]:
        for H in [30]:
            # outdir = f'__results/sumprod/E{E}/H{H}/lr{lr}/iters{num_iters}'
            outdir = f'__results/mouse_transcripts/E{E}/H{H}/lr{lr}/iters{num_iters}'
            os.makedirs(outdir, exist_ok=True)
            print(f'seq2seq '
                  f'--architecture rnn+mlp '
                  f'--data-file ../data/mouse-genome/seqs_transformed.txt '
                  f'--config ./lang_config_transcripts.json '
                  f'--bidirectional '
                  f'--batch-size 16 '
                  f'--print-loss-interval 2 '
                  f'--embedding-dim {E} '
                  f'--hidden-size {H} '
                  f'--num-iters {num_iters} '
                  f'--learning-rate {lr} '
                  f'--device cuda:{idx % 8} '
                  f'-o {outdir} ')
            idx += 1
