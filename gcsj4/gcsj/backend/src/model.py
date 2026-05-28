import torch
import torchvision.transforms as transforms
from torchvision import models, ResNet50_Weights
from PIL import Image
import os

class DogBreedModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.breed_list = self._load_breed_list()
    
    def _load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), '../models/dog_breed_model.pth')
        
        if os.path.exists(model_path):
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 120)
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 120)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_breed_list(self):
        return [
            'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
            'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
            'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
            'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
            'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres',
            'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
            'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow',
            'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont',
            'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter',
            'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
            'french_bulldog', 'german_shepherd', 'german_short-haired_pointer',
            'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane',
            'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound',
            'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound',
            'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
            'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg',
            'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mastiff', 'miniature_pinscher',
            'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
            'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound',
            'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone',
            'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed',
            'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
            'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
            'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle',
            'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier',
            'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner',
            'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet',
            'wire-haired_fox_terrier', 'yorkshire_terrier'
        ]
    
    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = image.convert('RGB')
        image = transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def predict(self, file):
        image = Image.open(file.stream)
        image_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        results = []
        for i in range(5):
            breed_idx = top5_idx[0][i].item()
            breed_name = self.breed_list[breed_idx]
            confidence = top5_prob[0][i].item() * 100
            results.append({
                'breed': breed_name.replace('_', ' '),
                'confidence': round(confidence, 2)
            })
        
        return {'predictions': results}
    
    def get_breed_list(self):
        return [breed.replace('_', ' ') for breed in self.breed_list]