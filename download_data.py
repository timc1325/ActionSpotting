import logging
from SoccerNet.Downloader import SoccerNetDownloader

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def download_action_spotting_labels(local_dir="data/"):
    """
    Downloads action spotting labels (Labels-v2.json) for the train, validation, and test splits
    from the SoccerNet dataset and stores them in the specified local directory.

    Args:
        local_dir (str): Path to store the downloaded data. Defaults to "data/".
    """
    logger.info(f"Initializing SoccerNetDownloader with local directory: {local_dir}")
    downloader = SoccerNetDownloader(LocalDirectory=local_dir)

    logger.info("Starting download of action spotting labels (Labels-v2.json)...")
    downloader.downloadGames(
        files=["Labels-v2.json"],
        split=["train", "valid", "test"]
    )
    logger.info("Download completed successfully.")

if __name__ == "__main__":
    download_action_spotting_labels()
