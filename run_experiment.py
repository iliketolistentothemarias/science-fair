import argparse
import os
import sys
from src.preprocessing import run_preprocessing
from src.segmentation import process_rois
from src.radiomics_features import process_batch_radiomics
from src.deep_features import process_batch_deep
from src.model import load_and_merge_data, train_evaluate_pipeline
from src.utils import setup_logging, ensure_dir

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="NSCLC EGFR Prediction Pipeline")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--run_all", action="store_true", help="Run full pipeline")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--segment", action="store_true", help="Run ROI generation")
    parser.add_argument("--extract_radiomics", action="store_true", help="Run PyRadiomics")
    parser.add_argument("--extract_deep", action="store_true", help="Run Deep Features")
    parser.add_argument("--train", action="store_true", help="Run Training")
    parser.add_argument("--clinical_file", type=str, default="data/clinical.csv", help="Path to clinical data CSV")
    
    args = parser.parse_args()
    
    raw_dir = os.path.join(args.data_dir, "raw")
    processed_dir = os.path.join(args.data_dir, "processed")
    rois_dir = os.path.join(args.data_dir, "rois")
    features_dir = os.path.join(args.data_dir, "features")
    ensure_dir(features_dir)
    
    radiomics_csv = os.path.join(features_dir, "radiomics.csv")
    deep_csv = os.path.join(features_dir, "deep_features.csv")
    
    if args.run_all or args.preprocess:
        logger.info("Starting Preprocessing...")
        run_preprocessing(raw_dir, processed_dir)
        
    if args.run_all or args.segment:
        logger.info("Starting ROI Generation...")
        # Iterating over processed files to generate peritumoral from corresponding tumor masks
        # Assuming we have tumor masks in rois_dir named {PatientID}_tumor.nii.gz
        # And processed images in processed_dir named {PatientID}.nii.gz
        
        # User needs to provide tumor masks in rois_dir first!
        # We process them to create peritumoral.
        masks = [f for f in os.listdir(rois_dir) if f.endswith('_tumor.nii.gz')]
        for m in masks:
            pid = m.replace('_tumor.nii.gz', '')
            mask_path = os.path.join(rois_dir, m)
            process_rois(mask_path, rois_dir, pid)
            
    if args.run_all or args.extract_radiomics:
        logger.info("Starting Radiomics Extraction...")
        process_batch_radiomics(processed_dir, rois_dir, radiomics_csv)
        
    if args.run_all or args.extract_deep:
        logger.info("Starting Deep Feature Extraction...")
        process_batch_deep(processed_dir, rois_dir, deep_csv)
        
    if args.run_all or args.train:
        logger.info("Starting Training...")
        if not os.path.exists(args.clinical_file):
            logger.error(f"Clinical file not found at {args.clinical_file}. Cannot train.")
            return

        df = load_and_merge_data(radiomics_csv, deep_csv, args.clinical_file)
        if df is not None:
             train_evaluate_pipeline(df)

if __name__ == "__main__":
    main()
