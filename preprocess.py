import torch
from dataset import Vocabulary
import os
import transformer.Constants as Constants
from collections import Counter
import numpy as np


def read_instances_from_file(file, max_sent_len, keep_case=True):
    instances = []
    trimmed_sent_cnt = 0
    with open(file, 'r') as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_cnt += 1
                words = words[:max_sent_len]

            if words:
                instances += [[Constants.BOS_WORD] + words + [Constants.EOS_WORD]]

    print('[Info] Get {} instances from {}'.format(len(instances), file))

    if trimmed_sent_cnt > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_cnt, max_sent_len))

    return instances


def read_instances_from_dir(data_dir, max_sent_len, keep_case=True):
    instances = []
    for file in os.listdir(data_dir):
        sents = read_instances_from_file(file, max_sent_len, keep_case)
        instances += sents

    return instances


def build_vocab(instances, min_word_cnt=5):
    vocab = Vocabulary()
    all_words = [word for sent in instances for word in sent]
    full_vocab = Counter(all_words).most_common()  # [('a', 5), ('b', 4), ('c', 3)]
    print('[Info] Original Vocabulary size =', len(full_vocab))

    for item in full_vocab:
        if item[1] >= min_word_cnt:
            vocab.add_word(item[0])
        else:
            break

    print('[Info] Trimmed vocabulary size = {},'.format(len(vocab)),
          'each with minimum occurrence = {}'.format(min_word_cnt))

    print("[Info] Ignored word count = {}".format(len(full_vocab) - len(vocab)))

    return vocab


def tokenize(instances, vocab):
    '''convert words to index'''
    return [[vocab.get_index(word) for word in sent] for sent in instances]


