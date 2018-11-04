from transformer.Constants import *
from torch.utils.data import Dataset
from preprocess import tokenize

class Vocabulary():

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.__vocab_size = 0
        self.add_word(PAD_WORD)
        self.add_word(UNK_WORD)
        self.add_word(BOS_WORD)
        self.add_word(EOS_WORD)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = self.__vocab_size
            self.__vocab_size += 1

    def __len__(self):
        return self.__vocab_size

    def get_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[UNK_WORD]

    def get_word(self, idx):
        return self.idx2word[idx]


class InstanceDataset(Dataset):
    def __init__(self,instances_ids):
        self.instances_ids=instances_ids    # already converted


    def __len__(self):
        return len(self.instances_ids)

    def __getitem__(self, idx):
        return self.instances_ids[idx]