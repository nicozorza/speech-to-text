import argparse
from src.utils.ProjectData import ProjectData
from src.utils.word_low_pass_filter import ler, wer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure WER and LER between two files. Each line must contain a sentence.')
    parser.add_argument('-p', '--predictions', help='Path to predictions file', required=True)
    parser.add_argument('-t', '--truth', help='Path to ground truth file', required=True)
    args = vars(parser.parse_args())

    project_data = ProjectData()

    predictions_file = open(args['predictions'], 'r')
    targets_file = open(args['truth'], 'r')

    predictions_list = list(map(lambda x: x.replace('\n', ''), predictions_file.readlines()))
    predictions_file.close()
    targets_list = list(map(lambda x: x.replace('\n', ''), targets_file.readlines()))
    targets_file.close()

    acum_wer = 0
    acum_ler = 0
    for i in range(len(predictions_list)):
        l = ler(predictions_list[i], targets_list[i])
        w = wer(predictions_list[i], targets_list[i])

        print(f"{i} of {len(predictions_list)}. LER: {l}, WER: {w}")
        print(f"Truth: {targets_list[i]}")
        print(f"Pred: {predictions_list[i]}")
        print()

        acum_ler += l
        acum_wer += w

    print("Final results")
    print(f"LER: {acum_ler / len(predictions_list)}")
    print(f"WER: {acum_wer / len(predictions_list)}")






