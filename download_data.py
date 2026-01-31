import os
import sys
import argparse
import requests
import zipfile
import io
import pandas as pd
from src.utils import setup_logging, ensure_dir

logger = setup_logging()

def download_tcia_data(collection="NSCLC-Radiogenomics", output_dir="data/raw"):
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
    pass
    logger.info(f"Starting download for collection: {collection}")
    
    # Get series info
    try:
        series_data = nbia.getSeries(collection=collection)
        if not series_data:
            logger.error("No series found. Check collection name or internet connection.")
            return
            
        logger.info(f"Found {len(series_data)} series. Downloading...")
        
        # Download series
        # input_type="list" expects a list of SeriesInstanceUIDs or a dataframe
        # getSeries returns a list of dictionaries.
        
        # We can pass the dataframe directly if we convert it, 
        # or tcia_utils might handle the list of dicts or we extract SeriesInstanceUID.
        
        # Let's extract dataframe to be safe and use native download
        df = pd.DataFrame(series_data)
        
        # Filter for CT only if needed, but radiogenomics usually has the relevant ones.
        # The dataset description says "CT scans", so we might want to filter Modality=='CT'.
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
