import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Throughput (Tokens per Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    # single_mean, single_std = 14.878658342361451, 0.4937598559748656
    # device0_mean, device0_std =  8.11140751838684, 0.131766627349889
    # device1_mean, device1_std =  7.674647617340088, 0.3380200251461661
    
    # single_mean, single_std = 14.878658342361451, 0.4937598559748656
    # device0_mean, device0_std =  250032.67317468533, 1935.3901021944096
    # device1_mean, device1_std =  250067.13472861797, 1967.2256581495997
    # plot([device0_mean, device1_mean, single_mean],
    #     [device0_std, device1_std, single_std],
    #     ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
    #     'ddp_vs_rn_time.png')

    # pp_mean, pp_std = 39.665109634399414, 0.1530299186706543
    # mp_mean, mp_std = 49.35502254962921, 0.23672902584075928
    # pp_mean, pp_std = 16135.327297607066, 62.25087606806483
    # mp_mean, mp_std = 12967.570291512937, 62.19833613781793
    pp_mean, pp_std = 500099.807903, 3902.61576034
    mp_mean, mp_std = 254300.83295833477, 2658.329639366154
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Data Parallel - 2 GPUs', 'Single GPU'],
        'dd_vs_rn_tokens.png')