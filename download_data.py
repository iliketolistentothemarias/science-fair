import os
import sys
import argparse
import requests
import zipfile
import io
import pandas as pd
from src.utils import setup_logging, ensure_dir

logger = setup_logging()

def download_tcia_data(collection="NSCLC Radiogenomics", output_dir="data/raw", limit=None):
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
        # Create DataFrame from series info
        df = pd.DataFrame(series_data)
        
        # Filter for CT and SEG scans
        if 'Modality' in df.columns:
            # We want the CT and the Segmentations
            filtered_series = df[df['Modality'].isin(['CT', 'SEG'])]
            logger.info(f"Filtered to {len(filtered_series)} series (CT and SEG).")
        else:
            filtered_series = df
            
        if filtered_series.empty:
            logger.error("No CT or SEG series found in this collection.")
            return

        # Extract SeriesInstanceUIDs as a list
        uids = filtered_series['SeriesInstanceUID'].tolist()
        
        if limit and limit > 0:
            # We want to make sure we get pairs if possible, but for a trial, 
            # just taking the first N is okay. 
            # Better: if limit is 20, we take 20 CTs and their matching SEGs if we can find them.
            # For simplicity, let's just take the first N series.
            logger.info(f"Limiting download to the first {limit} series.")
            uids = uids[:limit]
        
        logger.info(f"Downloading {len(uids)} series to {output_dir}...")
        
        # Using input_type="list" is usually more reliable across tcia-utils versions
        nbia.downloadSeries(uids, path=output_dir, input_type="list")
        
        logger.info("Download completed.")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description="Download NSCLC-Radiogenomics Data")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of series to download")
    args = parser.parse_args()
    
    download_tcia_data(output_dir=args.output_dir, limit=args.limit)

if __name__ == "__main__":
    main()
