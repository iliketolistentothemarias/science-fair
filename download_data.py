import os
import sys
import argparse
import requests
import zipfile
import io
import pandas as pd
from src.utils import setup_logging, ensure_dir

logger = setup_logging()

def download_tcia_data(collection="NSCLC Radiogenomics", output_dir="data/raw"):
    """
    Downloads the NSCLC-Radiogenomics dataset from TCIA using tcia_utils.
    Note: Requires 'tcia-utils' package.
    """
    try:
        from tcia_utils import nbia
    except ImportError:
        logger.error("tcia-utils not installed. Please run: pip install tcia-utils")
        return

    ensure_dir(output_dir)
    
    # Try the provided name, and then try common variations if it fails
    variations = [collection]
    if "-" in collection:
        variations.append(collection.replace("-", " "))
    elif " " in collection:
        variations.append(collection.replace(" ", "-"))
        
    series_data = None
    for coll_name in variations:
        logger.info(f"Checking for collection: {coll_name}")
        try:
            series_data = nbia.getSeries(collection=coll_name)
            if series_data and len(series_data) > 0:
                logger.info(f"Found {len(series_data)} series for '{coll_name}'.")
                break
        except Exception as e:
            logger.debug(f"Failed attempt for {coll_name}: {e}")
            
    if not series_data:
        logger.error(f"No series found for any variation of '{collection}'. Check internet connection or TCIA status.")
        return
        
    logger.info(f"Starting download...")
    
    try:
        # Download series
        df = pd.DataFrame(series_data)
        
        if 'Modality' in df.columns:
            ct_series = df[df['Modality'] == 'CT']
            logger.info(f"Filtered to {len(ct_series)} CT series.")
        else:
            ct_series = df
            
        nbia.downloadSeries(series_data=ct_series, path=output_dir, input_type="pandas")
        
        logger.info("Download completed.")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download NSCLC-Radiogenomics Data")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    download_tcia_data(output_dir=args.output_dir)

if __name__ == "__main__":
    main()
