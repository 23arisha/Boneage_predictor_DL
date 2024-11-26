import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import numpy as np

# Define the model class
class BoneAgePretrainedNet(nn.Module):
    def __init__(self):
        super(BoneAgePretrainedNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.base_model(x)

# Load the pre-trained model
model = BoneAgePretrainedNet()
model.load_state_dict(torch.load('rsna_boneage_model.pth', map_location=torch.device('cpu')))


# Image prediction function
def predict_image(model, image_path):
    img = Image.open(image_path)  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor
    model.eval()
    
    with torch.no_grad():
        output = model(img_tensor)
    return output.item()  # Return the predicted bone age

# Streamlit UI
st.title("Bone Age Prediction")
st.write("Upload an image for bone age prediction")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Get prediction
    prediction = predict_image(model, uploaded_file)
    
    # Display the result
    st.write(f"Predicted Bone Age: {prediction:.2f} months")

