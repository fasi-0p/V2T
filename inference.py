import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from train_v2t import V2TModel, extract_frames  # Import definitions from training script

CONFIG = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

def generate_caption(video_path, model_path='v2t_model.pt'):
    checkpoint = torch.load(model_path, map_location=CONFIG['device'])
    config = checkpoint['config']

    model = V2TModel(config['embed_size'], config['hidden_size'], config['num_clusters'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(CONFIG['device'])
    model.eval()

    frames = extract_frames(video_path, config['num_frames'])
    if frames is None:
        return "Error: Could not extract frames"

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    frames_tensor = torch.stack([transform(frame) for frame in frames])
    frames_tensor = frames_tensor.to(CONFIG['device'])

    description = model.generate_description(frames_tensor)
    return description

if __name__ == "__main__":
    video_path = r'C:\Users\FASI OWAIZ AHMED\Desktop\v2t\data\a.avi'
    caption = generate_caption(video_path, 'v2t_model.pt')
    print(f"Generated Caption: {caption}")