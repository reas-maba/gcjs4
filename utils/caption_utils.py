import torch
import nltk


def caption_to_tensor(caption, vocab, max_len=20):
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    
    if len(caption) < max_len:
        caption.extend([vocab('<pad>')] * (max_len - len(caption)))
    else:
        caption = caption[:max_len]
    
    return torch.tensor(caption, dtype=torch.long)


def tensor_to_caption(tensor, vocab):
    caption = []
    for idx in tensor:
        word = vocab.idx2word[idx.item()]
        if word == '<end>':
            break
        if word != '<start>':
            caption.append(word)
    return ' '.join(caption)


def generate_caption(image, encoder, decoder, vocab, transform, device='cpu'):
    image = transform(image).unsqueeze(0).to(device)
    
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        features = encoder(image)
        sampled_ids = decoder.sample(features)
        sampled_ids = torch.tensor(sampled_ids, dtype=torch.long).unsqueeze(0)
        
        caption = tensor_to_caption(sampled_ids[0], vocab)
    
    return caption
