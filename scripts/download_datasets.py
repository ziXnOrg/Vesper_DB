#!/usr/bin/env python3
"""
Download standard ANN benchmark datasets for testing Vesper.
Datasets are downloaded in HDF5 format from Hugging Face and ann-benchmarks.
"""

import os
import sys
import h5py
import numpy as np
import requests
from pathlib import Path
from typing import Tuple, Dict, Any
import hashlib
from tqdm import tqdm

# Dataset URLs and metadata
DATASETS = {
    "sift-128-euclidean": {
        "url": "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift-128-euclidean.hdf5",
        "size_mb": 525,
        "dimensions": 128,
        "num_vectors": 1000000,
        "metric": "euclidean"
    },
    "fashion-mnist-784-euclidean": {
        "url": "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/fashion-mnist-784-euclidean.hdf5",
        "size_mb": 228,
        "dimensions": 784,
        "num_vectors": 60000,
        "metric": "euclidean"
    },
    "glove-100-angular": {
        "url": "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/glove-100-angular.hdf5",
        "size_mb": 485,
        "dimensions": 100,
        "num_vectors": 1183514,
        "metric": "angular"
    },
    "glove-25-angular": {
        "url": "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/glove-25-angular.hdf5",
        "size_mb": 127,
        "dimensions": 25,
        "num_vectors": 1183514,
        "metric": "angular"
    },
    "mnist-784-euclidean": {
        "url": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        "size_mb": 217,
        "dimensions": 784,
        "num_vectors": 60000,
        "metric": "euclidean"
    }
}

def download_file(url: str, filepath: Path, expected_size_mb: int) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0:
            total_size = expected_size_mb * 1024 * 1024
        
        block_size = 8192
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=filepath.name) as pbar:
                for chunk in response.iter_content(block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Downloaded {filepath.name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False

def verify_hdf5_dataset(filepath: Path, metadata: Dict[str, Any]) -> bool:
    """Verify the downloaded HDF5 file contains expected data."""
    try:
        with h5py.File(filepath, 'r') as f:
            # Check standard ANN benchmark format
            if 'train' not in f and 'test' not in f:
                print(f"✗ Invalid format: missing 'train' or 'test' datasets")
                return False
            
            # Verify dimensions if train data exists
            if 'train' in f:
                train_shape = f['train'].shape
                if len(train_shape) != 2:
                    print(f"✗ Invalid train shape: {train_shape}")
                    return False
                
                if train_shape[1] != metadata['dimensions']:
                    print(f"✗ Dimension mismatch: expected {metadata['dimensions']}, got {train_shape[1]}")
                    return False
                
                print(f"  Train vectors: {train_shape[0]:,} x {train_shape[1]}D")
            
            # Verify test/query data
            if 'test' in f:
                test_shape = f['test'].shape
                print(f"  Test vectors: {test_shape[0]:,} x {test_shape[1]}D")
            
            # Check for ground truth
            if 'neighbors' in f:
                neighbors_shape = f['neighbors'].shape
                print(f"  Ground truth: {neighbors_shape[0]:,} queries x {neighbors_shape[1]} neighbors")
            
            print(f"✓ Verified {filepath.name}")
            return True
            
    except Exception as e:
        print(f"✗ Failed to verify {filepath.name}: {e}")
        return False

def convert_to_fvecs(hdf5_path: Path, output_dir: Path) -> bool:
    """Convert HDF5 dataset to FVECS format for compatibility."""
    try:
        with h5py.File(hdf5_path, 'r') as f:
            base_name = hdf5_path.stem
            
            # Convert train data
            if 'train' in f:
                train_data = f['train'][:]
                fvecs_path = output_dir / f"{base_name}_base.fvecs"
                write_fvecs(fvecs_path, train_data)
                print(f"  Converted train → {fvecs_path.name}")
            
            # Convert test data
            if 'test' in f:
                test_data = f['test'][:]
                fvecs_path = output_dir / f"{base_name}_query.fvecs"
                write_fvecs(fvecs_path, test_data)
                print(f"  Converted test → {fvecs_path.name}")
            
            # Convert ground truth
            if 'neighbors' in f:
                neighbors = f['neighbors'][:]
                ivecs_path = output_dir / f"{base_name}_groundtruth.ivecs"
                write_ivecs(ivecs_path, neighbors.astype(np.int32))
                print(f"  Converted neighbors → {ivecs_path.name}")
            
            if 'distances' in f:
                distances = f['distances'][:]
                fvecs_path = output_dir / f"{base_name}_distances.fvecs"
                write_fvecs(fvecs_path, distances)
                print(f"  Converted distances → {fvecs_path.name}")
            
            return True
            
    except Exception as e:
        print(f"✗ Failed to convert {hdf5_path.name}: {e}")
        return False

def write_fvecs(filepath: Path, data: np.ndarray):
    """Write numpy array to FVECS format."""
    with open(filepath, 'wb') as f:
        for vec in data:
            dim = np.array([len(vec)], dtype=np.int32)
            dim.tofile(f)
            vec.astype(np.float32).tofile(f)

def write_ivecs(filepath: Path, data: np.ndarray):
    """Write numpy array to IVECS format."""
    with open(filepath, 'wb') as f:
        for vec in data:
            dim = np.array([len(vec)], dtype=np.int32)
            dim.tofile(f)
            vec.astype(np.int32).tofile(f)

def main():
    # Setup directories
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    hdf5_dir = data_dir / "hdf5"
    fvecs_dir = data_dir / "fvecs"
    
    # Create directories
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    fvecs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Vesper Dataset Downloader")
    print(f"=" * 50)
    print(f"Data directory: {data_dir}")
    print()
    
    # Select datasets to download
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
    else:
        print("Available datasets:")
        for i, (name, info) in enumerate(DATASETS.items(), 1):
            print(f"  {i}. {name} ({info['dimensions']}D, {info['size_mb']}MB)")
        print()
        print("Enter dataset names or numbers (space-separated), or 'all':")
        choice = input("> ").strip()
        
        if choice.lower() == 'all':
            selected = list(DATASETS.keys())
        else:
            selected = []
            for item in choice.split():
                if item.isdigit():
                    idx = int(item) - 1
                    if 0 <= idx < len(DATASETS):
                        selected.append(list(DATASETS.keys())[idx])
                else:
                    if item in DATASETS:
                        selected.append(item)
    
    if not selected:
        print("No datasets selected.")
        return
    
    print(f"\nDownloading {len(selected)} dataset(s)...")
    print()
    
    # Download and verify datasets
    for name in selected:
        if name not in DATASETS:
            print(f"✗ Unknown dataset: {name}")
            continue
        
        info = DATASETS[name]
        hdf5_path = hdf5_dir / f"{name}.hdf5"
        
        print(f"Dataset: {name}")
        print(f"  Dimensions: {info['dimensions']}")
        print(f"  Vectors: {info['num_vectors']:,}")
        print(f"  Metric: {info['metric']}")
        
        # Check if already downloaded
        if hdf5_path.exists():
            print(f"  Already downloaded: {hdf5_path}")
            if verify_hdf5_dataset(hdf5_path, info):
                # Optionally convert to FVECS
                print(f"  Converting to FVECS format...")
                convert_to_fvecs(hdf5_path, fvecs_dir)
            print()
            continue
        
        # Download
        print(f"  Downloading {info['size_mb']}MB...")
        if download_file(info['url'], hdf5_path, info['size_mb']):
            # Verify
            if verify_hdf5_dataset(hdf5_path, info):
                # Convert to FVECS
                print(f"  Converting to FVECS format...")
                convert_to_fvecs(hdf5_path, fvecs_dir)
        print()
    
    print("Done!")
    print(f"HDF5 files: {hdf5_dir}")
    print(f"FVECS files: {fvecs_dir}")

if __name__ == "__main__":
    # Check dependencies
    try:
        import h5py
        import requests
        from tqdm import tqdm
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install h5py requests tqdm")
        sys.exit(1)
    
    main()