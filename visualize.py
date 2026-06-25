import json
import matplotlib.pyplot as plt
import os


def plot_loss_curve(logs, save_path='outputs/loss_curve.png'):
    plt.figure(figsize=(10, 6))
    
    plt.plot(logs['train_loss'], label='Training Loss')
    plt.plot(logs['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_perplexity_curve(logs, save_path='outputs/perplexity_curve.png'):
    plt.figure(figsize=(10, 6))
    
    plt.plot(logs['train_perplexity'], label='Training Perplexity')
    plt.plot(logs['val_perplexity'], label='Validation Perplexity')
    
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main():
    log_path = 'logs/training_log.json'
    
    if not os.path.exists(log_path):
        print("Creating mock training logs for visualization...")
        logs = {
            'train_loss': [4.5, 3.8, 3.2, 2.8, 2.5, 2.2, 2.0, 1.8, 1.6, 1.5],
            'val_loss': [4.2, 3.6, 3.1, 2.8, 2.6, 2.4, 2.3, 2.2, 2.1, 2.0],
            'train_perplexity': [90.0, 45.0, 24.5, 16.5, 12.2, 9.1, 7.4, 6.1, 5.0, 4.5],
            'val_perplexity': [66.7, 36.7, 22.2, 16.4, 13.5, 11.1, 10.0, 9.1, 8.2, 7.4]
        }
        
        os.makedirs('logs', exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=4)
    else:
        with open(log_path, 'r') as f:
            logs = json.load(f)
    
    plot_loss_curve(logs)
    plot_perplexity_curve(logs)
    
    print("Loss curve saved to outputs/loss_curve.png")
    print("Perplexity curve saved to outputs/perplexity_curve.png")


if __name__ == '__main__':
    main()
