import torch
import torch.nn as nn
import torchvision.models.video as models
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from .utils import setup_logging, ensure_dir

logger = setup_logging()

class MedicalResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MedicalResNet, self).__init__()
        # Load standard 3D ResNet-18 pretrained on Kinetics-400
        self.resnet = models.r3d_18(pretrained=pretrained)
        
        # Modify the first layer to accept 1 channel (CT) instead of 3 (RGB)
        # Average the weights of the RGB channels to initialize the single channel
        old_conv = self.resnet.stem[0]
        new_conv = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            
        self.resnet.stem[0] = new_conv
        
        # Remove classification head to get features (Global Average Pooling is just before fc)
        # In torchvision r3d_18, 'avgpool' is the last layer before 'fc'.
        # We want the output of avgpool flattened.
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        # x shape: (Batch, 1, Depth, Height, Width)
        x = self.resnet(x)
        return x

def extract_deep_features(image_path, mask_path, patient_id, model, device):
    """
    Extracts deep features from the masked tumor region.
    """
    try:
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # Apply mask to image (zero out non-tumor regions)
        # Using the core tumor mask is standard for deep features, 
        # but prompt says "features from the tumor core". 
        # It implies DL features are from the core? 
        # "Can combining deep learning features ... with radiomic texture analysis of the peritumoral microenvironment"
        # "Second stream will use a 3D ResNet-18 ... to extract 512 deep features that capture volumetric spatial patterns."
        # It doesn't explicitly restrict DL to core, but standard practice is usually core or core+margin. 
        # I'll stick to Core for DL as implied by "segmentation stream" separation usually seen in papers.
        
        # Convert to numpy
        img_arr = sitk.GetArrayFromImage(image)
        mask_arr = sitk.GetArrayFromImage(mask)
        
        # Zero out background
        img_arr = img_arr * mask_arr
        
        # Crop to bbox to reduce computation and focus network
        # Find bbox
        coords = np.argwhere(mask_arr > 0)
        if coords.size == 0:
            logger.warning(f"Empty mask for {patient_id}")
            return None
            
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0) + 1
        
        img_crop = img_arr[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Normalize/Scale if needed (Standardization already done? No, just resampling)
        # Simple min-max or z-score on the crop can help DL
        if img_crop.std() > 0:
            img_crop = (img_crop - img_crop.mean()) / img_crop.std()
            
        # Convert to Tensor
        # Add channel and batch dims: (1, 1, D, H, W)
        tensor = torch.from_numpy(img_crop).float().unsqueeze(0).unsqueeze(0)
        
        # Handle size constraints? 
        # ResNet usually expects some minimum size, but Conv3d is flexible.
        # If too small, might crash. Padding might be needed.
        # Padded to at least 16x16x16?
        
        # Resize/Interpolate to fixed size? Or keep variable?
        # Variable size works with GAP, but batching requires fixed size.
        # Here we process one by one, so variable size is okay-ish, 
        # but huge volumes run OOM. 
        # Let's resize to a fixed volume like 64x64x64 or 128x128x128 for consistency.
        # Or simple center crop/pad.
        # Let's resize for simplicity and robustness.
        
        target_size = (64, 64, 64)
        tensor = torch.nn.functional.interpolate(tensor, size=target_size, mode='trilinear', align_corners=False)
        
        tensor = tensor.to(device)
        
        with torch.no_grad():
            features = model(tensor)
            
        # Features shape: (1, 512)
        return features.cpu().numpy().flatten()
        
    except Exception as e:
        logger.error(f"Deep feature extraction failed for {patient_id}: {e}")
        return None

def process_batch_deep(data_dir, rois_dir, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = MedicalResNet().to(device)
    model.eval()
    
    all_features = []
    
    images = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    
    for img_file in images:
        patient_id = img_file.replace('.nii.gz', '')
        image_path = os.path.join(data_dir, img_file)
        # Deep features from Tumor Core
        tumor_mask = os.path.join(rois_dir, f"{patient_id}_tumor.nii.gz")
        
        if os.path.exists(tumor_mask):
            feats = extract_deep_features(image_path, tumor_mask, patient_id, model, device)
            
            if feats is not None:
                row = {'PatientID': patient_id}
                for i, f in enumerate(feats):
                    row[f"deep_{i}"] = f
                all_features.append(row)
                
    if all_features:
        df = pd.DataFrame(all_features)
        ensure_dir(os.path.dirname(output_file))
        df.to_csv(output_file, index=False)
        logger.info(f"Saved deep features to {output_file}")
    else:
        logger.warning("No deep features extracted.")
