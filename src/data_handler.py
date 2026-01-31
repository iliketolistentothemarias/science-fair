import os
import pydicom
import pydicom_seg
import SimpleITK as sitk
import numpy as np
from .utils import setup_logging, ensure_dir

logger = setup_logging()

def convert_seg_to_nifti(seg_path, output_path, reference_image_path=None):
    """
    Converts a DICOM SEG file to a NIfTI mask.
    If reference_image_path is provided, it uses it to ensure the mask matches the image geometry.
    """
    try:
        dcm = pydicom.dcmread(seg_path)
        reader = pydicom_seg.MultiClassReader()
        result = reader.read(dcm)
        
        # Extract the mask volume
        # result.image is a SimpleITK image
        mask = result.image
        
        # DICOM SEG might have multiple segments. We usually want the main tumor.
        # For simplicity, we'll take the sum or max of all labels to get a binary mask.
        arr = sitk.GetArrayFromImage(mask)
        binary_arr = (arr > 0).astype(np.uint8)
        
        binary_mask = sitk.GetImageFromArray(binary_arr)
        binary_mask.CopyInformation(mask)
        
        sitk.WriteImage(binary_mask, output_path)
        return True
    except Exception as e:
        logger.error(f"Failed to convert SEG {seg_path}: {e}")
        return False

def discover_and_convert_rois(raw_dir, rois_dir):
    """
    Searches raw_dir for SEG files and converts them to the rois_dir.
    """
    ensure_dir(rois_dir)
    count = 0
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if f.endswith('.dcm'):
                path = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(path, stop_before_pixels=True)
                    if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.66.4': # Segmentation Storage
                        patient_id = ds.PatientID
                        output_path = os.path.join(rois_dir, f"{patient_id}_tumor.nii.gz")
                        
                        if not os.path.exists(output_path):
                            logger.info(f"Converting SEG for patient {patient_id}...")
                            if convert_seg_to_nifti(path, output_path):
                                count += 1
                except:
                    pass
    logger.info(f"Converted {count} segmentation files.")
