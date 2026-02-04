#!/usr/bin/env python3
"""
Reverse testing experiment for 2024 Northern Hemisphere season.
Model: fluProfiler_v0_1
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))

import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from fluprofiler.data import load_data, preprocess_sequences
from fluprofiler.models import fluProfiler_v0_1
from fluprofiler.evaluation import calculate_metrics, bootstrap_confidence_interval
from fluprofiler.utils import setup_logger, save_json


def load_config(config_path):
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run reverse testing experiment."""

    # Load configuration
    config_path = Path(__file__).parent / 'config.yaml'
    config = load_config(config_path)

    # Setup logging
    logger = setup_logger(
        name=config['experiment']['name'],
        log_file=f"logs/{config['experiment']['name']}.log"
    )

    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")

    try:
        # Load and preprocess data
        logger.info("Loading data...")
        data = load_data('data/processed/antigenic_data.csv')

        # Split data by season
        test_season = config['data']['test_season']
        train_seasons = config['data']['train_seasons']

        train_data = data[data['season'].isin(train_seasons)]
        test_data = data[data['season'] == test_season]

        logger.info(f"Train seasons: {train_seasons}")
        logger.info(f"Test season: {test_season}")
        logger.info(f"Train samples: {len(train_data)}")
        logger.info(f"Test samples: {len(test_data)}")

        # Load model
        logger.info("Loading model...")
        model_config = config['model']
        model = fluProfiler_v0_1.from_pretrained(model_config['checkpoint'])

        # Make predictions
        logger.info("Making predictions...")
        model.eval()

        # Preprocess test data for model input
        test_sequences = preprocess_sequences(test_data)

        with torch.no_grad():
            predictions = model(test_sequences).cpu().numpy().flatten()

        actual_values = test_data['hi_distance'].values

        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = calculate_metrics(actual_values, predictions, print_result=False)

        # Bootstrap confidence intervals
        logger.info("Calculating confidence intervals...")
        mae_ci = bootstrap_confidence_interval(
            np.abs(actual_values - predictions),
            np.mean,
            n_resamples=config['evaluation']['bootstrap_iterations']
        )

        metrics['mae_ci_lower'] = mae_ci[0]
        metrics['mae_ci_upper'] = mae_ci[1]

        # Save results
        output_dir = Path(config['evaluation']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        save_json(metrics, output_dir / 'metrics.json')

        # Save predictions if requested
        if config['evaluation']['save_predictions']:
            results_df = pd.DataFrame({
                'actual': actual_values,
                'predicted': predictions,
                'season': test_data['season'],
                'strain_pair': test_data['strain_pair']
            })
            results_df.to_csv(output_dir / 'predictions.csv', index=False)

        # Log results
        logger.info("Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")

        logger.info(f"Results saved to {output_dir}")
        logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()