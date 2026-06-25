import torch
import pytest

from model import EncoderCNN, DecoderRNN


class TestEncoderCNN:
    def test_encoder_initialization(self):
        encoder = EncoderCNN(embed_size=256)
        assert encoder is not None
        assert hasattr(encoder, 'resnet')
        assert hasattr(encoder, 'embed')
    
    def test_encoder_forward(self):
        encoder = EncoderCNN(embed_size=256)
        encoder.eval()
        
        dummy_image = torch.randn(1, 3, 224, 224)
        output = encoder(dummy_image)
        
        assert output.shape == (1, 256)
        assert not torch.any(torch.isnan(output))


class TestDecoderRNN:
    def test_decoder_initialization(self):
        decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=1000)
        assert decoder is not None
        assert hasattr(decoder, 'embedding')
        assert hasattr(decoder, 'lstm')
        assert hasattr(decoder, 'linear')
    
    def test_decoder_forward(self):
        decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=1000)
        decoder.eval()
        
        dummy_features = torch.randn(2, 256)
        dummy_captions = torch.randint(0, 1000, (2, 10))
        
        output = decoder(dummy_features, dummy_captions)
        
        assert output.shape == (2, 10, 1000)
        assert not torch.any(torch.isnan(output))
    
    def test_decoder_sample(self):
        decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=1000)
        decoder.eval()
        
        dummy_features = torch.randn(1, 256)
        sampled_ids = decoder.sample(dummy_features)
        
        assert isinstance(sampled_ids, list)
        assert len(sampled_ids) > 0
        assert all(isinstance(id, int) for id in sampled_ids)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
