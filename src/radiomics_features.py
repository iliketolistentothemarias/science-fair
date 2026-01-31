import os
import pandas as pd
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from .utils import setup_logging, ensure_dir

logger = setup_logging()

def get_radiomics_extractor():
    """
    Configures and returns a PyRadiomics feature extractor.
    """
    params = {}
    params['binWidth'] = 25
    params['resampledPixelSpacing'] = None # We already resampled
    params['interpolator'] = 'sitkLinear'
    params['verbose'] = True
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.enableAllImageTypes()
    extractor.disableAllFeatures()
    
    # Enable features based on proposal
    extractor.enableFeatureClass('firstorder')
    extractor.enableFeatureClass('shape')
    extractor.enableFeatureClass('glcm')
    extractor.enableFeatureClass('glrlm')
    extractor.enableFeatureClass('glszm') # Often useful
    extractor.enableFeatureClass('gldm')  # Often useful
    extractor.enableFeatureClass('ngtdm') # Often useful
    
    return extractor

def extract_features(image_path, mask_path, patient_id, region_name="tumor"):
    """
    Extracts features for a single image/mask pair.
    """
    extractor = get_radiomics_extractor()
    
    try:
        logger.info(f"Extracting {region_name} features for {patient_id}...")
        result = extractor.execute(image_path, mask_path)
        
        # Clean up result keys
        clean_result = {}
        for key, value in result.items():
            if not key.startswith("diagnostics"):
                feature_name = f"{region_name}_{key}"
                clean_result[feature_name] = value
        
        clean_result['PatientID'] = patient_id
        return clean_result
        
    except Exception as e:
        logger.error(f"Radiomics extraction failed for {patient_id} ({region_name}): {e}")
        return None

def process_batch_radiomics(data_dir, rois_dir, output_file):
    """
    Batch processes all patients and saves a CSV.
    """
    # Assuming standard project structure
    # data_dir has images: {PatientID}.nii.gz
    # rois_dir has masks: {PatientID}_tumor.nii.gz, {PatientID}_peritumoral.nii.gz
    
    all_features = []
    
    # Find all patients
    images = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    
    for img_file in images:
        patient_id = img_file.replace('.nii.gz', '')
        image_path = os.path.join(data_dir, img_file)
        
        # Tumor Core
        tumor_mask = os.path.join(rois_dir, f"{patient_id}_tumor.nii.gz") # Naming convention assumption
        if os.path.exists(tumor_mask):
            feats = extract_features(image_path, tumor_mask, patient_id, "core")
            if feats:
                # Peritumoral
                pt_mask = os.path.join(rois_dir, f"{patient_id}_peritumoral.nii.gz")
                if os.path.exists(pt_mask):
                    pt_feats = extract_features(image_path, pt_mask, patient_id, "peritumoral")
                    if pt_feats:
                        # Merge dictionaires
                        # Remove PatientID from one to avoid dupe or just let pandas handle it (it will just overwrite)
                        feats.update(pt_feats)
                        all_features.append(feats)
    
    if all_features:
        df = pd.DataFrame(all_features)
        ensure_dir(os.path.dirname(output_file))
        df.to_csv(output_file, index=False)
        logger.info(f"Saved radiomics features to {output_file}")
    else:
        logger.warning("No features extracted.")
