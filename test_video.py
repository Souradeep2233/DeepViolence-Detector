import os 
import numpy as np 
import pandas as pd
import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset, random_split
from glob import glob
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.models.video import swin3d_t

# sample testing video
# dir="V_101.mp4"
# Tranforms for testing
transforms_val=T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),  
])

# Frame Extraction         
def extract_frames(video,frame_count):
    # ~ 30 FPS => !s has 30 frames, so we're considering 16 frames , ~=50% of a frame
    # That is >.5s (a moment) , rather than entire frame (redundant)
    vid=cv2.VideoCapture(video)
    total_frames=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # For higher probability to detext cpture from monment , we 
    # select evenly spaced frames in time series. (Uniform Sampling)

    frame_indices=list(torch.linspace(0,total_frames-1,steps=frame_count).long().numpy())
    frames=[]
    
    for i in range(total_frames):
        ret, frame = vid.read() # captures each frame
        if not ret:
            break
        if i in frame_indices: # if frame ind considered
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frames.append(frame)
            
    vid.release()
    return frames

class VDC(nn.Module):
    def __init__(self,num_classes=1):
        super(VDC,self).__init__()

        self.backbone=swin3d_t(progress=True)

        #Freeze the backbone
        for params in self.backbone.parameters():
            params.requires_grad=False
            
        #By default all layers are requires_grad= True : Train the head
        # small head
        self.backbone.head=nn.Sequential(
            # 768 is a dimensional vector , not feature map alike in CNNs
            # [b,t,c,h,w]=====[4,16,224,224]------->[4,16,768,7,7]--->Avg.Pooling---->[768*1] vector dim
            nn.Linear( 768, 192), 
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Linear(192,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,num_classes)
            
        ) # Freezing weights except the classifier head...-> Will only train the head .

    def forward(self,x):
        return self.backbone(x).squeeze(1)

device= "cuda" if torch.cuda.is_available() else "cpu"
model=VDC().to(device)
model.load_state_dict(torch.load("VD_classification.pt",weights_only=True))
model.eval()

frames=extract_frames(dir,16)

processed_frames=[transforms_val(x) for x in frames]
video_tensor=torch.stack(processed_frames).to(device)
video_tensor=video_tensor.unsqueeze(0)

video_tensor=video_tensor.permute(0,2,1,3,4).to(device) #[c,t,h,w]
output = model(video_tensor)
if output>.5:
    print("Violent")
else:
    print("Non Violent")
