
import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Loading ResNet50 model...")
    model = models.resnet50(pretrained=True)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

try:
    logger.info("Loading ImageNet classes...")
    with open('imagenet_classes.json', 'r') as f:
        imagenet_classes = json.load(f)
    logger.info("Classes loaded successfully")
except Exception as e:
    logger.error(f"Failed to load classes: {e}")
    raise

DOG_CLASS_START = 151
DOG_CLASS_END = 268

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_dog_breed(image):
    if image is None:
        return "请上传一张狗的图片", {}
    
    try:
        if isinstance(image, Image.Image):
            img = image
        else:
            img = Image.fromarray(image)
        
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        dog_results = []
        for i in range(DOG_CLASS_START, DOG_CLASS_END + 1):
            dog_results.append((i, probabilities[i].item()))
        
        dog_results.sort(key=lambda x: x[1], reverse=True)
        
        top_5 = dog_results[:5]
        
        result_dict = {}
        for idx, prob in top_5:
            breed_name = imagenet_classes[str(idx)]
            confidence = prob * 100
            result_dict[breed_name] = confidence
        
        if not result_dict:
            return "未识别到狗品种", {}
        
        top_breed = max(result_dict, key=result_dict.get)
        top_confidence = result_dict[top_breed]
        
        return f"识别结果：{top_breed} (置信度: {top_confidence:.2f}%)", result_dict
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"识别过程中发生错误: {e}", {}

with gr.Blocks(title="狗品种识别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐕 DogBreeds Recognizer V1.0")
    gr.Markdown("上传一张狗的图片，系统将自动识别狗的品种")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传图片")
            submit_btn = gr.Button("开始识别", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="识别结果", placeholder="识别结果将显示在这里...")
            output_plot = gr.BarChart(label="置信度排行", x_label="品种", y_label="置信度(%)")
    
    submit_btn.click(
        fn=predict_dog_breed,
        inputs=input_image,
        outputs=[output_text, output_plot]
    )
    
    gr.Markdown("### 支持识别的犬种")
    gr.Markdown("系统支持识别118种常见犬种，包括：金毛寻回犬、拉布拉多、哈士奇、贵宾犬、柴犬等")

if __name__ == "__main__":
    logger.info("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
    logger.info("App closed")
