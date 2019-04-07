from typing import List


def levenshtein(a, b):
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]


def word_low_pass_filter(sentence: str, vocab_list: List[str]) -> str:
    word_list = sentence.split(' ')
    filtered_word_list = []
    for word in word_list:
        filtered_word = min(vocab_list, key=lambda t: levenshtein(word, t))
        filtered_word_list.append(filtered_word)

    return " ".join(filtered_word_list)


def ler(prediction: str, truth: str) -> float:
    max_len = max(len(truth), len(prediction))
    return levenshtein(truth, prediction) / max_len


def wer(prediction: str, truth: str) -> float:
    prediction = prediction.split(' ')
    truth = truth.split(' ')
    max_len = max(len(truth), len(prediction))
    return levenshtein(truth, prediction) / max_len
