import SimpleITK as sitk
import numpy as np
from .utils import setup_logging

logger = setup_logging()

def generate_peritumoral_region(tumor_mask, dilation_radius_mm=2.0):
    """
    Generates the peritumoral region by dilating the tumor mask and performing a subtraction.
    
    Args:
        tumor_mask (sitk.Image): Binary mask of the tumor (Int type).
        dilation_radius_mm (float): Radius of dilation in millimeters.
        
    Returns:
        sitk.Image: Binary mask of the peritumoral region.
    """
    
    # Ensure mask is binary
    tumor_mask = sitk.Cast(tumor_mask, sitk.sitkUInt8)
    
    # Create the dilation filter
    # For binary dilation, we can use BinaryDilateImageFilter with a kernel radius.
    # The kernel radius is in pixels, so we need to convert mm to pixels.
    # Assuming isotropic 1mm spacing as per preprocessing step.
    
    spacing = tumor_mask.GetSpacing()
    
    # Calculate radius in pixels (assuming roughly isotropic, otherwise need anisotropic kernel)
    # We will take the max spacing to be conservative or average. 
    # For 1x1x1mm spacing, radius is just the mm value.
    radius_pixels = [int(np.ceil(dilation_radius_mm / s)) for s in spacing]
    
    dilator = sitk.BinaryDilateImageFilter()
    dilator.SetKernelRadius(radius_pixels)
    dilator.SetKernelType(sitk.sitkBall)
    dilator.SetForegroundValue(1)
    
    dilated_mask = dilator.Execute(tumor_mask)
    
    # Subtract original tumor to get the ring
    # Peritumoral = Dilated - Original
    
    peritumoral_mask = sitk.Xor(dilated_mask, tumor_mask)
    
    # Mask out any region outside the lung if we had a lung mask, but prompt doesn't specify using lung mask for this.
    # It just says "subtracting the original tumor core".
    
    return peritumoral_mask

def process_rois(mask_path, output_dir, patient_id):
    """
    Loads a mask, generates the peritumoral region, and saves both.
    """
    try:
        mask = sitk.ReadImage(mask_path)
        
        # Ensure it's 3D
        if mask.GetDimension() != 3:
            logger.error(f"Mask for {patient_id} is not 3D.")
            return

        peritumoral = generate_peritumoral_region(mask, dilation_radius_mm=2.0)
        
        # Save
        pt_path = f"{output_dir}/{patient_id}_peritumoral.nii.gz"
        sitk.WriteImage(peritumoral, pt_path)
        logger.info(f"Generated peritumoral mask for {patient_id} at {pt_path}")
        
    except Exception as e:
        logger.error(f"Failed to process mask for {patient_id}: {e}")
