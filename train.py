import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import nltk

from model import EncoderCNN, DecoderRNN
from data import CocoDataset, build_vocab, save_vocab, load_vocab, get_train_transform, get_val_transform
from utils import caption_to_tensor, Logger

nltk.download('punkt')


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    images = torch.stack(images, 0)
    
    caption_lengths = [len(caption) for caption in captions]
    padded_captions = torch.zeros(len(captions), max(caption_lengths)).long()
    
    for i, cap in enumerate(captions):
        end = caption_lengths[i]
        padded_captions[i, :end] = cap[:end]
    
    return images, padded_captions, caption_lengths


def train_epoch(encoder, decoder, dataloader, criterion, optimizer, device):
    encoder.train()
    decoder.train()
    
    total_loss = 0.0
    total_samples = 0
    
    for images, captions, _ in dataloader:
        images = images.to(device)
        captions = captions.to(device)
        
        optimizer.zero_grad()
        
        features = encoder(images)
        outputs = decoder(features, captions)
        
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def val_epoch(encoder, decoder, dataloader, criterion, device):
    encoder.eval()
    decoder.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, captions, _ in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            
            features = encoder(images)
            outputs = decoder(features, captions)
            
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    train_dir = 'data/train2014'
    val_dir = 'data/val2014'
    train_annotation = 'data/annotations/captions_train2014.json'
    val_annotation = 'data/annotations/captions_val2014.json'
    vocab_path = 'data/vocab.pkl'
    
    if os.path.exists(vocab_path):
        vocab = load_vocab(vocab_path)
    else:
        with open(train_annotation, 'r') as f:
            train_annotations = json.load(f)
        vocab = build_vocab(train_annotations)
        save_vocab(vocab, vocab_path)
    
    vocab_size = len(vocab)
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = CocoDataset(train_dir, train_annotation, train_transform)
    val_dataset = CocoDataset(val_dir, val_annotation, val_transform)
    
    def custom_collate(data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)
        
        images = torch.stack(images, 0)
        
        caption_tensors = []
        for cap in captions:
            cap_tensor = caption_to_tensor(cap, vocab)
            caption_tensors.append(cap_tensor)
        
        caption_lengths = [len(ct[ct != 0]) for ct in caption_tensors]
        max_len = max(caption_lengths)
        
        padded_captions = torch.zeros(len(caption_tensors), max_len).long()
        for i, ct in enumerate(caption_tensors):
            end = min(len(ct), max_len)
            padded_captions[i, :end] = ct[:end]
        
        return images, padded_captions, caption_lengths
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    logger = Logger('logs')
    
    print(f"Training started on {device}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    for epoch in range(num_epochs):
        train_loss, train_perplexity = train_epoch(encoder, decoder, train_loader, criterion, optimizer, device)
        val_loss, val_perplexity = val_epoch(encoder, decoder, val_loader, criterion, device)
        
        logger.add_entry(train_loss, val_loss, train_perplexity, val_perplexity)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}")
        
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'vocab_size': vocab_size
        }, f'outputs/model_epoch_{epoch+1}.pth')
    
    logger.save()
    print("Training completed.")


if __name__ == '__main__':
    main()
