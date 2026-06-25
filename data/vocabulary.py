import nltk
import pickle
import os
from collections import Counter


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)


def build_vocab(annotations, threshold=5):
    counter = Counter()
    
    for ann in annotations['annotations']:
        caption = ann['caption']
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
    
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    
    return vocab


def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
