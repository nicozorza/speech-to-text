import argparse
import sys
sys.path.append('.')
from src.utils.sentence_utils import ler, wer
import matplotlib.pyplot as plt
from scipy import stats


def evaluate_metrics(predictions_list, targets_list, show_graphics):
    error_list = []

    acum_wer = 0
    acum_ler = 0
    for i in range(len(predictions_list)):
        l = ler(predictions_list[i], targets_list[i])
        w = wer(predictions_list[i], targets_list[i])

        print(f"{i} of {len(predictions_list)}. LER: {l}, WER: {w}")
        print(f"Truth: {targets_list[i]}")
        print(f"Pred: {predictions_list[i]}")
        print()

        error_list.append({
            'len': len(targets_list[i]),
            'ler': l,
            'wer': w
        })

        acum_ler += l
        acum_wer += w

    print("Final results")
    print(f"LER: {acum_ler / len(predictions_list)}")
    print(f"WER: {acum_wer / len(predictions_list)}")

    if show_graphics:
        lengths = list(map(lambda x: x['len'], error_list))
        values = list(map(lambda x: x['ler'], error_list))
        bin_means, bin_edges, binnumber = stats.binned_statistic(lengths, values, statistic='mean', bins=100)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2
        plt.plot(bin_centers, bin_means)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure WER and LER between two files. Each line must contain a sentence.')
    parser.add_argument('-p', '--predictions', help='Path to predictions file', required=True)
    parser.add_argument('-t', '--truth', help='Path to ground truth file', required=True)
    parser.add_argument('-g', '--graphics', help='Plot results', required=False)
    args = vars(parser.parse_args())

    predictions_file = open(args['predictions'], 'r')
    targets_file = open(args['truth'], 'r')

    predictions_list = list(map(lambda x: x.replace('\n', ''), predictions_file.readlines()))
    predictions_file.close()
    targets_list = list(map(lambda x: x.replace('\n', ''), targets_file.readlines()))
    targets_file.close()

    evaluate_metrics(predictions_list, targets_list, args.get('graphics'))
