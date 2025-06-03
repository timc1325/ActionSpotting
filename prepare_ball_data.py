"""
prepare_ball_data.py

Download and unzip SoccerNet Ball Action Spotting 2024 data.
"""

# Adapted from: https://github.com/recokick/ball-action-spotting/blob/master/download_action_data.py

import os
import zipfile
import argparse
import logging
from SoccerNet.Downloader import SoccerNetDownloader

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def unzip_and_cleanup(split, data_dir):
    """
    Unzips and removes the zip file for a given split.
    """
    zip_path = os.path.join(data_dir, f"{split}.zip")

    if not os.path.isfile(zip_path):
        logger.warning(f"Zip file {zip_path} does not exist. Skipping.")
        return

    logger.info(f"Unzipping {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        logger.info(f"Successfully unzipped {zip_path}")
    except zipfile.BadZipFile:
        logger.error(f"Failed to unzip {zip_path}. Corrupted file?")
        return

    logger.info(f"Removing {zip_path}...")
    try:
        os.remove(zip_path)
        logger.info(f"{zip_path} removed.")
    except Exception as e:
        logger.error(f"Failed to delete {zip_path}: {e}")

def main(args):
    list_splits = ["train", "valid", "test", "challenge"]
    subtask_data_dir = os.path.join(args.dataset_dir, "spotting-ball-2024")

    logger.info("Initializing SoccerNet downloader...")
    downloader = SoccerNetDownloader(LocalDirectory=args.dataset_dir)

    logger.info("Downloading zipped splits...")
    downloader.downloadDataTask(
        task="spotting-ball-2024",
        split=list_splits,
        password=args.password_videos
    )

    logger.info("Unzipping and cleaning up...")
    for split in list_splits:
        unzip_and_cleanup(split, subtask_data_dir)

    logger.info("All splits processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare SoccerNet Ball Action Spotting 2024 dataset.")
    parser.add_argument("--dataset_dir", type=str, default="data/", help="Path to dataset output directory.")
    parser.add_argument("--password_videos", type=str, default="s0cc3rn3t", help="NDA-protected password for video download.")

    args = parser.parse_args()
    main(args)
