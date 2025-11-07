import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#configurations, 
CONFIG = {
    'train_dir': 'C:\Users\FASI OWAIZ AHMED\Desktop\v2t\data', #dataset directory
    'model_save_path': 'v2t_model.pt',
    'num_epochs': 1,
    'batch_size': 4,
    'learning_rate': 0.001,
    'embed_size': 512,
    'hidden_size': 512,
    'num_frames': 16,
    'num_clusters': 100,  # Number of visual pattern clusters
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

#video preprocessing
def extract_frames(video_path, num_frames=16): #extract linearly spaced frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    return np.array(frames[:num_frames])

def get_video_files(directory):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    video_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    return video_files

#dataset
class VideoDataset(Dataset):
    def __init__(self, video_dir, num_frames=16):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.video_files = get_video_files(video_dir)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_name)
        
        frames = extract_frames(video_path, self.num_frames)
        if frames is None:
            frames = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
        
        frames_tensor = torch.stack([self.transform(frame) for frame in frames])
        
        return frames_tensor, video_name

# ======================== MODEL ARCHITECTURE ========================
class FrameEncoder(nn.Module):
    def __init__(self, embed_size):
        super(FrameEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def forward(self, frames):
        batch_size, num_frames, c, h, w = frames.shape
        frames = frames.view(batch_size * num_frames, c, h, w)
        
        with torch.no_grad():
            features = self.resnet(frames)
        
        features = features.view(batch_size * num_frames, -1)
        features = self.linear(features)
        features = self.bn(features)
        features = features.view(batch_size, num_frames, -1)
        
        return features

class TemporalEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return context, lstm_out

class VideoDescriptor(nn.Module):
    def __init__(self, hidden_size, num_clusters):
        super(VideoDescriptor, self).__init__()
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_clusters)
        )
        
        self.scene_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_clusters)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_clusters)
        )
    
    def forward(self, context):
        motion = self.motion_head(context)
        scene = self.scene_head(context)
        action = self.action_head(context)
        
        return motion, scene, action

class V2TModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_clusters):
        super(V2TModel, self).__init__()
        self.frame_encoder = FrameEncoder(embed_size)
        self.temporal_encoder = TemporalEncoder(embed_size, hidden_size)
        self.descriptor = VideoDescriptor(hidden_size, num_clusters)
        
        # Predefined vocabulary for description generation
        self.motion_words = [
            'static', 'slow', 'fast', 'moving', 'panning', 'zooming', 'shaking', 'stable',
            'rotating', 'tracking', 'following', 'aerial', 'handheld', 'smooth', 'dynamic',
            'walking', 'running', 'jumping', 'dancing', 'driving', 'flying', 'swimming'
        ]
        
        self.scene_words = [
            'indoor', 'outdoor', 'urban', 'nature', 'beach', 'mountain', 'forest', 'street',
            'room', 'office', 'kitchen', 'park', 'garden', 'city', 'building', 'sky',
            'water', 'field', 'night', 'day', 'sunset', 'sunrise', 'cloudy', 'sunny',
            'dark', 'bright', 'crowded', 'empty', 'modern', 'vintage'
        ]
        
        self.action_words = [
            'person', 'people', 'group', 'animal', 'cat', 'dog', 'bird', 'car', 'vehicle',
            'bike', 'playing', 'working', 'talking', 'sitting', 'standing', 'eating',
            'drinking', 'cooking', 'reading', 'writing', 'sports', 'music', 'performance',
            'game', 'celebration', 'event', 'activity', 'nature', 'landscape'
        ]
    
    def forward(self, frames):
        features = self.frame_encoder(frames)
        context, temporal_features = self.temporal_encoder(features)
        motion, scene, action = self.descriptor(context)
        
        return motion, scene, action, context
    
    def generate_description(self, frames):
        self.eval()
        with torch.no_grad():
            motion_logits, scene_logits, action_logits, _ = self.forward(frames.unsqueeze(0))
            
            # Get top predictions
            motion_probs = torch.softmax(motion_logits, dim=1)[0]
            scene_probs = torch.softmax(scene_logits, dim=1)[0]
            action_probs = torch.softmax(action_logits, dim=1)[0]
            
            # Get top-k predictions
            motion_top = torch.topk(motion_probs, min(3, len(self.motion_words)))
            scene_top = torch.topk(scene_probs, min(3, len(self.scene_words)))
            action_top = torch.topk(action_probs, min(3, len(self.action_words)))
            
            # Build description
            motion_desc = [self.motion_words[idx % len(self.motion_words)] 
                          for idx in motion_top.indices.cpu().numpy()]
            scene_desc = [self.scene_words[idx % len(self.scene_words)] 
                         for idx in scene_top.indices.cpu().numpy()]
            action_desc = [self.action_words[idx % len(self.action_words)] 
                          for idx in action_top.indices.cpu().numpy()]
            
            # Create natural language description
            description = f"A {motion_desc[0]} video showing {action_desc[0]} in a {scene_desc[0]} {scene_desc[1]} setting"
            
            return description

# ======================== TRAINING ========================
def train_model(config):
    print(f"Device: {config['device']}")
    
    # Get video files
    video_files = get_video_files(config['train_dir'])
    print(f"Found {len(video_files)} videos in directory")
    
    if len(video_files) == 0:
        print("ERROR: No video files found!")
        return
    
    # Create dataset
    dataset = VideoDataset(config['train_dir'], config['num_frames'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                           shuffle=True, num_workers=2)
    
    # Initialize model
    model = V2TModel(config['embed_size'], config['hidden_size'], config['num_clusters'])
    model = model.to(config['device'])
    
    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=config['learning_rate'])
    
    # Self-supervised loss: consistency and diversity
    print("Starting training...")
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        for i, (frames, video_names) in enumerate(dataloader):
            frames = frames.to(config['device'])
            
            # Forward pass
            motion, scene, action, context = model(frames)
            
            # Self-supervised losses
            # 1. Encourage diversity in predictions (entropy)
            motion_entropy = -torch.mean(torch.sum(
                torch.softmax(motion, dim=1) * torch.log_softmax(motion, dim=1), dim=1))
            scene_entropy = -torch.mean(torch.sum(
                torch.softmax(scene, dim=1) * torch.log_softmax(scene, dim=1), dim=1))
            action_entropy = -torch.mean(torch.sum(
                torch.softmax(action, dim=1) * torch.log_softmax(action, dim=1), dim=1))
            
            # 2. Encourage confident predictions (negative entropy bonus)
            entropy_loss = -(motion_entropy + scene_entropy + action_entropy)
            
            # 3. Context consistency (features should be normalized)
            consistency_loss = torch.mean((torch.norm(context, dim=1) - 1.0) ** 2)
            
            # Total loss
            loss = entropy_loss + 0.1 * consistency_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}], "
                      f"Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Average Loss: {avg_loss:.4f}")
        
        # Sample output every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_frames = frames[0]
                description = model.generate_description(sample_frames)
                print(f"Sample description: {description}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, config['model_save_path'])
    print(f"\nModel saved to {config['model_save_path']}")

# ======================== INFERENCE ========================
def generate_caption(video_path, model_path='v2t_model.pt'):
    """Generate caption for a video"""
    checkpoint = torch.load(model_path, map_location=CONFIG['device'])
    config = checkpoint['config']
    
    # Initialize model
    model = V2TModel(config['embed_size'], config['hidden_size'], config['num_clusters'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config['device'])
    model.eval()
    
    # Extract frames
    print(f"Processing video: {video_path}")
    frames = extract_frames(video_path, config['num_frames'])
    
    if frames is None:
        return "Error: Could not extract frames from video"
    
    # Transform frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frames_tensor = torch.stack([transform(frame) for frame in frames])
    frames_tensor = frames_tensor.to(config['device'])
    
    # Generate description
    description = model.generate_description(frames_tensor)
    
    return description

# ======================== MAIN EXECUTION ========================
if __name__ == "__main__":
    MODE = 'train'  # Change to 'inference' for caption generation
    
    if MODE == 'train':
        # TRAINING - Just provide video directory!
        train_model(CONFIG)
        
    elif MODE == 'inference':
        # INFERENCE - Provide single video path
        test_video_path = 'C:\Users\FASI OWAIZ AHMED\Desktop\v2t\data\a.avi'
        
        caption = generate_caption(test_video_path, CONFIG['model_save_path'])
        
        print(f"Generated Caption: {caption}")