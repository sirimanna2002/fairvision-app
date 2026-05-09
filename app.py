import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import gdown
import os


FILE_ID = "1eeKHQj2Zz2NJ6hdCwYLPA_t1TWvNTAzc"  

@st.cache_resource
def load_model():
    url = f"https://drive.google.com/drive/my-drive={FILE_ID}"
    model_path = "fairface_model_script.pt"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, model_path, quiet=False)
    
    # Load TorchScript model (no pickle issues!)
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Age groups
age_groups = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']


st.set_page_config(page_title="FairVision - Age Classifier", page_icon="👤")

st.title("👤 FairVision - Age Group Classification")
st.write("Upload a face image to predict age group")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=250)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
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
