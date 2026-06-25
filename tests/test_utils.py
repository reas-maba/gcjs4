import pytest
import torch

from utils import caption_to_tensor, tensor_to_caption
from data import Vocabulary


class TestCaptionUtils:
    def test_caption_to_tensor(self):
        vocab = Vocabulary()
        vocab.add_word('a')
        vocab.add_word('dog')
        vocab.add_word('is')
        vocab.add_word('running')
        
        caption = 'a dog is running'
        tensor = caption_to_tensor(caption, vocab)
        
        assert tensor.shape == (20,)
        assert tensor[0] == vocab('<start>')
        assert tensor[-1] == vocab('<pad>')
    
    def test_tensor_to_caption(self):
        vocab = Vocabulary()
        vocab.add_word('a')
        vocab.add_word('dog')
        vocab.add_word('is')
        vocab.add_word('running')
        
        tensor = torch.tensor([1, 4, 5, 6, 7, 2, 0, 0, 0, 0])
        
        caption = tensor_to_caption(tensor, vocab)
        
        assert caption == 'a dog is running'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
