import json
import os


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.logs = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': []
        }
        
        os.makedirs(log_dir, exist_ok=True)
    
    def add_entry(self, train_loss=None, val_loss=None, 
                  train_perplexity=None, val_perplexity=None):
        if train_loss is not None:
            self.logs['train_loss'].append(train_loss)
        if val_loss is not None:
            self.logs['val_loss'].append(val_loss)
        if train_perplexity is not None:
            self.logs['train_perplexity'].append(train_perplexity)
        if val_perplexity is not None:
            self.logs['val_perplexity'].append(val_perplexity)
    
    def save(self, filename='training_log.json'):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=4)
    
    def load(self, filename='training_log.json'):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'r') as f:
            self.logs = json.load(f)
        return self.logs
