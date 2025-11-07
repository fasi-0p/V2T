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

# ======================== CONFIG ========================
CONFIG = {
    'train_dir': r'C:\Users\FASI OWAIZ AHMED\Desktop\v2t\data',
    'model_save_path': 'v2t_model.pt',
    'num_epochs': 1,
    'batch_size': 4,
    'learning_rate': 0.001,
    'embed_size': 512,
    'hidden_size': 512,
    'num_frames': 16,
    'num_clusters': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ======================== DATA UTILS ========================
def extract_frames(video_path, num_frames=16):
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

    if not frames:
        return None

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array(frames[:num_frames])

def get_video_files(directory):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return [f for f in os.listdir(directory)
            if any(f.lower().endswith(ext) for ext in video_extensions)]

class VideoDataset(Dataset):
    def __init__(self, video_dir, num_frames=16):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.video_files = get_video_files(video_dir)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
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

# ======================== MODEL DEFINITIONS ========================
class FrameEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, frames):
        b, n, c, h, w = frames.shape
        frames = frames.view(b * n, c, h, w)
        with torch.no_grad():
            features = self.resnet(frames)
        features = features.view(b * n, -1)
        features = self.linear(features)
        features = self.bn(features)
        return features.view(b, n, -1)

class TemporalEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
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
        super().__init__()
        def head():
            return nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, num_clusters)
            )
        self.motion_head, self.scene_head, self.action_head = head(), head(), head()

    def forward(self, context):
        return self.motion_head(context), self.scene_head(context), self.action_head(context)

class V2TModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_clusters):
        super().__init__()
        self.frame_encoder = FrameEncoder(embed_size)
        self.temporal_encoder = TemporalEncoder(embed_size, hidden_size)
        self.descriptor = VideoDescriptor(hidden_size, num_clusters)

    def forward(self, frames):
        features = self.frame_encoder(frames)
        context, _ = self.temporal_encoder(features)
        motion, scene, action = self.descriptor(context)
        return motion, scene, action, context

# ======================== TRAINING LOOP ========================
def train_model(config):
    print(f"Device: {config['device']}")
    dataset = VideoDataset(config['train_dir'], config['num_frames'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    model = V2TModel(config['embed_size'], config['hidden_size'], config['num_clusters']).to(config['device'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate'])

    print("Starting training...")
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0

        for i, (frames, _) in enumerate(dataloader):
            frames = frames.to(config['device'])
            motion, scene, action, context = model(frames)

            motion_entropy = -torch.mean(torch.sum(
                torch.softmax(motion, dim=1) * torch.log_softmax(motion, dim=1), dim=1))
            scene_entropy = -torch.mean(torch.sum(
                torch.softmax(scene, dim=1) * torch.log_softmax(scene, dim=1), dim=1))
            action_entropy = -torch.mean(torch.sum(
                torch.softmax(action, dim=1) * torch.log_softmax(action, dim=1), dim=1))

            entropy_loss = -(motion_entropy + scene_entropy + action_entropy)
            consistency_loss = torch.mean((torch.norm(context, dim=1) - 1.0) ** 2)
            loss = entropy_loss + 0.1 * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Average Loss: {total_loss / len(dataloader):.4f}")

    torch.save({'model_state_dict': model.state_dict(), 'config': config}, config['model_save_path'])
    print(f"Model saved to {config['model_save_path']}")

if __name__ == "__main__":
    train_model(CONFIG)