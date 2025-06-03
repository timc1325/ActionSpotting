import logging
from SoccerNet.Evaluation.ActionSpotting import evaluate

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def evaluate_action_spotting(PATH_DATASET, PATH_PREDICTIONS):
    """
    Evaluates action spotting predictions using the SoccerNet evaluation tools.

    Args:
        PATH_DATASET (str): Path to the SoccerNet dataset.
        PATH_PREDICTIONS (str): Path to the folder containing the prediction JSON file.
    """
    logger.info("Starting evaluation with tight metric (version 2)...")
    results = evaluate(
        SoccerNet_path=PATH_DATASET,
        Predictions_path=PATH_PREDICTIONS,
        split="test",
        version=2,
        prediction_file="results_spotting.json",
        metric="tight"
    )

    logger.info(f"tight Average mAP: {results['a_mAP']:.4f}")
    logger.info(f"tight Average mAP per class: {results['a_mAP_per_class']}")
    logger.info(f"tight Average mAP visible: {results['a_mAP_visible']:.4f}")
    logger.info(f"tight Average mAP visible per class: {results['a_mAP_per_class_visible']}")
    logger.info(f"tight Average mAP unshown: {results['a_mAP_unshown']:.4f}")
    logger.info(f"tight Average mAP unshown per class: {results['a_mAP_per_class_unshown']}")

if __name__ == "__main__":
    # Replace with actual paths or parse from argparse
    PATH_DATASET = "data/"
    PATH_PREDICTIONS = "predictions/"
    evaluate_action_spotting(PATH_DATASET, PATH_PREDICTIONS)
