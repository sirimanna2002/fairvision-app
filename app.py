import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import gdown
import os
import numpy as np


class FairFaceCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

FILE_ID = "12Orm7hnQlJIWLmnA9PEp3WiK7uffgYBB"
URL = f"https://drive.google.com/drive/my-drive={FILE_ID}"

@st.cache_resource
def load_model():
    URL = f"https://drive.google.com/drive/my-drive={FILE_ID}"
    model_path = "best_fairface_model.pth"

    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
         gdown.download(URL, model_path, quiet=False)
        
    model = FairFaceCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Age group names
age_groups = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']


st.set_page_config(page_title="FairVision - Age Classifier", page_icon="👤")

st.title("👤 FairVision - Age Group Classification")
st.write("Upload a face image to predict age group")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=250)
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    # Predict
    model = load_model()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        top3_probs, top3_indices = torch.topk(probs, 3)
    
    st.write("### Predictions:")
    for i in range(3):
        idx = top3_indices[0][i].item()
        prob = top3_probs[0][i].item()
        st.write(f"**{i+1}. {age_groups[idx]}** - {prob*100:.1f}% confidence")
    
    st.caption("Note: Demo only. Not for official use.")
