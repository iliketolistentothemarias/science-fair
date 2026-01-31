import os
import glob
import numpy as np
import SimpleITK as sitk
from .utils import setup_logging, ensure_dir

logger = setup_logging()

def load_dicom_series(directory):
    """Reads a DICOM series from a directory and returns a SimpleITK image."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def resample_image(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear):
    """Resamples an image to a new isotropic spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(-1000) # Air for CT
    
    return resample.Execute(image)

def apply_lung_window(image, level=-600, width=1500):
    """Applies lung windowing to the CT scan."""
    # IntensityWindowingImageFilter maps intensities to [outputMin, outputMax]
    # We want to keep the HU values relevant for the window, but arguably
    # for deep learning we might want to normalize to [0, 1] or similar.
    # However, PyRadiomics expects original HU or consistent scaling.
    # Let's just clip to the window for now to remove outliers.
    
    lower_bound = level - width / 2
    upper_bound = level + width / 2
    
    threshold_filter = sitk.ClampImageFilter()
    threshold_filter.SetLowerBound(lower_bound)
    threshold_filter.SetUpperBound(upper_bound)
    return threshold_filter.Execute(image)

def process_patient(patient_dir, output_dir, patient_id):
    """Processes a single patient's DICOM series."""
    try:
        logger.info(f"Processing patient {patient_id}...")
        
        # Load
        image = load_dicom_series(patient_dir)
        
        # Resample
        image_resampled = resample_image(image)
        
        # Windowing (Optional: Can be done here or kept raw for radiomics)
        # For radiomics, we usually want the raw HU but resampled. 
        # For DL, we often window. 
        # Strategy: Save the resampled raw image. Windowing can be applied on the fly or saved separately.
        # Let's save the resampled raw image (better for pyradiomics).
        
        output_path = os.path.join(output_dir, f"{patient_id}.nii.gz")
        sitk.WriteImage(image_resampled, output_path)
        logger.info(f"Saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to process {patient_id}: {e}")

def run_preprocessing(raw_data_root, output_root):
    """Runs preprocessing for all patients in the raw data directory."""
    ensure_dir(output_root)
    # Assuming standard TCIA structure: PatientID / Study / Series / .dcm
    # We'll just look for directories that look like they contain DICOMs
    
    # This part depends heavily on the folder structure. 
    # For now, let's assume raw_data_root contains PatientIDs.
    
    patient_dirs = [d for d in os.listdir(raw_data_root) if os.path.isdir(os.path.join(raw_data_root, d))]
    
    for pid in patient_dirs:
        # Find the series folder (often deeply nested)
        # We'll walk effectively to find the first directory with multiple dicoms
        found_series = False
        for root, dirs, files in os.walk(os.path.join(raw_data_root, pid)):
            dicoms = [f for f in files if f.endswith('.dcm')]
            if len(dicoms) > 10: # Arbitrary threshold to find the scan volume
                process_patient(root, output_root, pid)
                found_series = True
                break
        
        if not found_series:
            logger.warning(f"No DICOM series found for {pid}")

if __name__ == "__main__":
    # Example usage
    RAW_DIR = "../data/raw"
    PROCESSED_DIR = "../data/processed"
    run_preprocessing(RAW_DIR, PROCESSED_DIR)
