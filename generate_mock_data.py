import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset
from src.utils import ensure_dir

def create_mock_dicom_series(output_dir, patient_id, num_slices=20, size=(128, 128)):
    """Creates a fake DICOM series."""
    series_dir = os.path.join(output_dir, patient_id, "Study1", "Series1")
    ensure_dir(series_dir)
    
    for i in range(num_slices):
        filename = os.path.join(series_dir, f"slice_{i:03d}.dcm")
        
        # Create Dummy DICOM
        ds = FileDataset(filename, {}, file_meta=pydicom.dataset.FileMetaDataset())
        ds.PatientID = patient_id
        ds.StudyInstanceUID = "1.2.3.4.5"
        ds.SeriesInstanceUID = "1.2.3.4.5.6"
        ds.SOPInstanceUID = f"1.2.3.4.5.6.{i}"
        ds.Modality = "CT"
        ds.Rows = size[0]
        ds.Columns = size[1]
        ds.PixelSpacing = [1.0, 1.0]
        ds.ImagePositionPatient = [0, 0, i * 1.0]
        ds.SliceLocation = i * 1.0
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        
        # Generate random noise + a "tumor" blob in the center
        arr = np.random.randint(-1000, -500, size, dtype=np.int16) # Lung air
        
        # Add a blob
        center = size[0] // 2
        radius = 10
        y, x = np.ogrid[:size[0], :size[1]]
        dist_sq = (x - center)**2 + (y - center)**2
        
        # Tumor slices in the middle
        if 5 <= i < 15:
            mask = dist_sq <= radius**2
            arr[mask] = np.random.randint(0, 100, size=np.count_nonzero(mask))
            
        ds.PixelData = arr.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        
        ds.save_as(filename)

def create_mock_tumor_mask(output_dir, patient_id, num_slices=20, size=(128, 128)):
    """Creates a corresponding tumor mask in processed/rois structure."""
    # We need to match the processed NIfTI dimensions. 
    # For simplicity, we create a NIfTI directly as if it was already available.
    
    # In the pipeline: 
    # 1. Raw DICOM -> Processed NIfTI (resampled)
    # 2. ROI (NIfTI) is expected to exist.
    
    # We will assume preprocessing has run or we manually create the processed NIfTI?
    # No, we want to test the full pipeline, so we create raw DICOMs first.
    # But for segmentation/feature extraction to work, we also need the ROIs.
    # In a real workflow, ROIs usually come from radiologists drawing on the original scan.
    # We need to simulate these "drawn" ROIs.
    # We'll save them as NIfTI in `data/rois` as expected by `run_experiment.py`.
    
    # Needs to match the resampled geometry.
    # Our preprocessing resamples to 1x1x1. Our mock DICOM is 1x1x1.
    # So dimensions should match.
    
    arr = np.zeros((num_slices, size[0], size[1]), dtype=np.uint8)
    
    # Draw same blob
    center = size[0] // 2
    radius = 10
    
    for i in range(num_slices):
        if 5 <= i < 15:
            y, x = np.ogrid[:size[0], :size[1]]
            dist_sq = (x - center)**2 + (y - center)**2
            mask = dist_sq <= radius**2
            arr[i, mask] = 1
            
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0, 0, 0))
    
    ensure_dir(output_dir)
    sitk.WriteImage(img, os.path.join(output_dir, f"{patient_id}_tumor.nii.gz"))

def generate_mock_data(root_dir, n_patients=5):
    raw_dir = os.path.join(root_dir, "raw")
    rois_dir = os.path.join(root_dir, "rois")
    
    ensure_dir(raw_dir)
    ensure_dir(rois_dir)
    
    clin_data = []
    
    for i in range(n_patients):
        pid = f"PAT_{i:03d}"
        create_mock_dicom_series(raw_dir, pid)
        create_mock_tumor_mask(rois_dir, pid)
        clin_data.append({'PatientID': pid, 'EGFR_Label': np.random.randint(0, 2)})
        
    pd.DataFrame(clin_data).to_csv(os.path.join(root_dir, "clinical.csv"), index=False)
    print(f"Generated mock data for {n_patients} patients.")

if __name__ == "__main__":
    generate_mock_data("data_mock")
