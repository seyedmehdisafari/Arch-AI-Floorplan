import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.dataset import RPlanDataset
from src.model import UNet
from PIL import Image
import os

DATA_DIR = "./data/Interface/static/Data/Img"
OUTPUT_DIR = "./assets" 

def test_and_save():
    # 1. Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 2. Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Load dataset and get one batch
    dataset = RPlanDataset(data_dir=DATA_DIR, img_size=(256, 256))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # batch_size=1 to process one by one
    
    # Get a random floor plan from the dataset
    inputs, targets = next(iter(dataloader))
    
    # 4. Load the trained U-Net model and its weights
    model = UNet(in_channels=1, out_channels=3).to(device)
    model.load_state_dict(torch.load("unet_rplan_latest.pth", map_location=device))
    model.eval() # Set model to evaluation mode

    # 5. Generate floor plan
    with torch.no_grad():
        inputs_dev = inputs.to(device)
        outputs = model(inputs_dev)
        outputs = outputs.cpu() # Bring back to CPU for saving as image

    # 6. Convert Tensors back to PIL Images to save them
    # PyTorch tensors are [C, H, W], we change them back to standard image format
    to_pil = transforms.ToPILImage()
    
    input_img = to_pil(inputs[0])
    target_img = to_pil(targets[0])
    output_img = to_pil(outputs[0])

    # 7. Save results side-by-side or individually
    input_img.save(os.path.join(OUTPUT_DIR, "1_input_walls.png"))
    target_img.save(os.path.join(OUTPUT_DIR, "2_ground_truth_rooms.png"))
    output_img.save(os.path.join(OUTPUT_DIR, "3_model_prediction.png"))

    print(f"\n🎉 Success! Check the results in the '{OUTPUT_DIR}' folder.")
    print("Files saved:")
    print(" - 1_input_walls.png (What the model saw)")
    print(" - 2_ground_truth_rooms.png (The original perfect plan)")
    print(" - 3_model_prediction.png (What your AI generated)")

if __name__ == "__main__":
    test_and_save()