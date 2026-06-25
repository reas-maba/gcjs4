import gradio as gr
import torch
import os
from PIL import Image

from model import EncoderCNN, DecoderRNN
from data import load_vocab, get_inference_transform
from utils import generate_caption


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 256
hidden_size = 512
vocab_path = 'data/vocab.pkl'
model_path = 'outputs/model_epoch_10.pth'


if not os.path.exists(vocab_path):
    print("Creating mock vocabulary...")
    from data import Vocabulary
    vocab = Vocabulary()
    common_words = ['a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                    'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                    'from', 'as', 'into', 'through', 'during', 'before', 'after',
                    'above', 'below', 'between', 'under', 'again', 'further', 'then',
                    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                    'just', 'but', 'if', 'or', 'because', 'until', 'while', 'about',
                    'against', 'he', 'she', 'it', 'they', 'we', 'you', 'i', 'this',
                    'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
                    'man', 'woman', 'boy', 'girl', 'dog', 'cat', 'car', 'tree', 'house',
                    'book', 'computer', 'phone', 'food', 'water', 'sun', 'moon', 'star',
                    'bird', 'fish', 'plant', 'flower', 'road', 'building', 'bridge',
                    'picture', 'photo', 'image', 'camera', 'person', 'people', 'group',
                    'walking', 'running', 'sitting', 'standing', 'playing', 'eating',
                    'drinking', 'talking', 'laughing', 'smiling', 'happy', 'sad', 'angry',
                    'beautiful', 'ugly', 'big', 'small', 'long', 'short', 'tall', 'wide',
                    'old', 'new', 'young', 'red', 'blue', 'green', 'yellow', 'black',
                    'white', 'brown', 'gray', 'purple', 'pink', 'orange']
    for word in common_words:
        vocab.add_word(word)
    from data import save_vocab
    save_vocab(vocab, vocab_path)

vocab = load_vocab(vocab_path)
vocab_size = len(vocab)

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
else:
    print("Using randomly initialized model for demonstration...")

encoder.eval()
decoder.eval()

transform = get_inference_transform()


def predict(image):
    caption = generate_caption(image, encoder, decoder, vocab, transform, device)
    return caption


with gr.Blocks(title="Image Captioning") as demo:
    gr.Markdown("# 图像描述系统 (Image Captioning)")
    gr.Markdown("上传一张图片，系统将自动生成英文描述。")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传图片")
            submit_btn = gr.Button("生成描述")
        
        with gr.Column():
            output_text = gr.Textbox(label="生成的描述", lines=3)
    
    submit_btn.click(fn=predict, inputs=input_image, outputs=output_text)


if __name__ == '__main__':
    demo.launch(debug=True)
