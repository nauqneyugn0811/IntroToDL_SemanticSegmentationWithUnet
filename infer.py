import torch
import argparse
from torchvision import transforms
from segmentation_models_pytorch import UnetPlusPlus
import cv2
import numpy as np

color_dict = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0)
}

def predict(model, image_path, output_path="output.png"):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    if image.shape[-1] == 3 and image[:, :, 0].all() == image[:, :, 2].all():
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image)
    prediction = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    colored_mask = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_dict.items():
        colored_mask[prediction == class_id] = color
    cv2.imwrite(output_path, colored_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script for image segmentation')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    model = UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    )
    checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predict(model, args.image_path)