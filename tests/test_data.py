import pytest
import torch

from data import Vocabulary, build_vocab, get_train_transform, get_val_transform


class TestVocabulary:
    def test_vocab_initialization(self):
        vocab = Vocabulary()
        assert len(vocab) == 4
        assert vocab('<pad>') == 0
        assert vocab('<start>') == 1
        assert vocab('<end>') == 2
        assert vocab('<unk>') == 3
    
    def test_vocab_add_word(self):
        vocab = Vocabulary()
        vocab.add_word('test')
        assert len(vocab) == 5
        assert vocab('test') == 4
        assert vocab.idx2word[4] == 'test'
    
    def test_vocab_unknown_word(self):
        vocab = Vocabulary()
        assert vocab('unknown') == 3


class TestTransforms:
    def test_train_transform(self):
        transform = get_train_transform()
        assert transform is not None
    
    def test_val_transform(self):
        transform = get_val_transform()
        assert transform is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
