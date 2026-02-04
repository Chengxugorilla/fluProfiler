"""
I/O utilities for fluProfiler.
"""

import os
import json
import pickle
import yaml
from pathlib import Path


def save_json(data, filepath, indent=4):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath: Path to save file
        indent: JSON indentation
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath):
    """
    Load data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data, filepath):
    """
    Save data to pickle file.

    Args:
        data: Data to save
        filepath: Path to save file
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    """
    Load data from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_yaml(data, filepath):
    """
    Save data to YAML file.

    Args:
        data: Data to save
        filepath: Path to save file
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(filepath):
    """
    Load data from YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(dir_path):
    """
    Ensure directory exists, create if necessary.

    Args:
        dir_path: Directory path
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def list_files(directory, extension=None):
    """
    List files in directory with optional extension filter.

    Args:
        directory: Directory path
        extension: File extension filter (e.g., '.json')

    Returns:
        List of file paths
    """
    files = []
    for file in os.listdir(directory):
        if extension and not file.endswith(extension):
            continue
        files.append(os.path.join(directory, file))
    return sorted(files)