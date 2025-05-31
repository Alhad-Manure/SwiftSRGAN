import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Resampling
import sys

from models import Generator

# ---------- CONFIGURATION ----------
IMAGE_PATH = r'D:\\Mtech\\Sem2\\Mini_Project\\SRGAN\\Training1\\Results\\Test1.jpg'       # Change to your input image
MODEL_PATH = 'model.pt'        # Change to your .pt model path
OUTPUT_PATH = r'D:\\Mtech\\Sem2\\Mini_Project\\SRGAN\\Training1\\Results\\Test1Output.JPG'     # Output image path
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------

# 1. Load and preprocess image
transform_input = transforms.Compose([
    #transforms.Resize((96, 96), interpolation=Image.BICUBIC),
    transforms.Resize(96, interpolation=Resampling.BICUBIC),
    transforms.ToTensor()
])

# Load image and preprocess
image = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = transform_input(image).unsqueeze(0)  # shape: [1, 3, 96, 96]

# 2. Load model
# model = torch.load(MODEL_PATH, map_location=DEVICE)
# model.eval()

model = Generator()
checkpoint = torch.load(r'D:\\Mtech\\Sem2\\Mini_Project\\SRGAN\\Training1\\Results\\netG_4x_epoch200.pth.tar', map_location=DEVICE)
model.load_state_dict(checkpoint['model'])
model.to(DEVICE)
model.eval()

# 3. Inference
with torch.no_grad():
    input_tensor = input_tensor.to(DEVICE)
    #The .cpu() call moves the tensor from the GPU (if it was on one) to the CPU. 
    #This is necessary because some operations—like converting a tensor to a PIL image or saving it require 
    # the tensor to be on the CPU, not the GPU.
    output_tensor = model(input_tensor).cpu().squeeze(0)  # shape: [3, H, W]

# 4. Postprocess output
transform_output = transforms.ToPILImage()
output_image = transform_output(output_tensor.clamp(0, 1))  # Clamp in case of out-of-range values

# Display the output image
output_image.show()  

# 5. Save output
output_image.save(OUTPUT_PATH)
print(f"Output saved to {OUTPUT_PATH}")
