import os


idx = 0
lrs = [0.001, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
for lr, num_iters in zip(lrs, [1000 for _ in lrs]):
    for E in [3]:
        for H in [50, 80]:
            # outdir = f'__results/sumprod/E{E}/H{H}/lr{lr}/iters{num_iters}'
            outdir = f'__results/prod/E{E}/H{H}/lr{lr}/iters{num_iters}'
            os.makedirs(outdir, exist_ok=True)
            print(f'seq2seq '
                  # f'--data-file ./tests/sumprod_num_seqs_10000_seq_len_1000_to_2000.csv '
                  f'--data-file ./tests/prod_num_seqs_10000_seq_len_1000_to_2000.csv '
                  f'--config ./tests/lang_config.json '
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
