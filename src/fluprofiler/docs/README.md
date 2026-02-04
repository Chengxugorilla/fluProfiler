# fluProfiler Documentation

## Overview

fluProfiler is a comprehensive toolkit for influenza virus antigenic characterization and vaccine design using deep learning approaches.

## Project Structure

```
src/fluprofiler/
├── data/              # Data loading, processing, and management
├── models/            # Model architectures and components
├── training/          # Training utilities and pipelines
├── evaluation/        # Performance evaluation and metrics
├── vaccine/           # Vaccine recommendation algorithms
├── sites/             # Key site analysis and interpretation
├── active_learning/   # Active learning implementations
├── utils/             # General utilities (logging, I/O, plotting)
├── experiments/       # Experiment scripts organized by type
├── notebooks/         # Exploratory analysis notebooks
├── configs/           # Configuration files
└── docs/              # Documentation
```

## Quick Start

### Installation

```bash
cd /path/to/fluProfiler
pip install -r requirements.txt
```

### Basic Usage

1. **Data Preparation**
   ```python
   from fluprofiler.data import load_data, preprocess_sequences

   # Load and preprocess data
   data = load_data('path/to/data.csv')
   processed_data = preprocess_sequences(data)
   ```

2. **Model Training**
   ```python
   from fluprofiler.models import fluProfiler_v0_1
   from fluprofiler.training import Trainer

   # Initialize model and trainer
   model = fluProfiler_v0_1(config)
   trainer = Trainer(model)

   # Train model
   history = trainer.train(train_loader, val_loader, epochs=100)
   ```

3. **Evaluation**
   ```python
   from fluprofiler.evaluation import calculate_metrics

   # Evaluate predictions
   metrics = calculate_metrics(actual_values, predictions)
   ```

## Experiment Types

### Reverse Testing
Scripts for evaluating model performance on historical data with reverse chronological splits.

- `experiments/reverse_test/2024NH/model4/`: 2024 Northern Hemisphere evaluation
- `experiments/reverse_test/2025SH/model4/`: 2025 Southern Hemisphere evaluation

### Extrapolation
Scripts for testing model extrapolation capabilities to future seasons.

### Ablation Studies
Scripts for ablation analysis to understand model components.

## Configuration

All experiments use YAML configuration files. Default settings are in `configs/default.yaml`.

## Key Features

- **Antigenic Distance Prediction**: Deep learning models for predicting antigenic distances between influenza strains
- **Vaccine Strain Selection**: Algorithms for optimal vaccine composition design
- **Active Learning**: Intelligent sample selection for efficient data labeling
- **Interpretability**: Tools for understanding model predictions and identifying key sites
- **Comprehensive Evaluation**: Statistical analysis including bootstrap confidence intervals

## API Reference

For detailed API documentation, see the docstrings in each module.

## Contributing

Please refer to the contributing guidelines in the main repository.