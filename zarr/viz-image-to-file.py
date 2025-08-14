#!/usr/bin/env python3
"""
Zarr Image Extractor

Usage:
    python3 viz-image-to-file.py <zarr_path> <image_dataset_path> [output_dir]

Example:
    python3 viz-image-to-file.py 1.zarr data/left_wrist_img
    python3 viz-image-to-file.py 1.zarr data/left_wrist_img my_output_folder

This will extract all images from the specified dataset in the Zarr store
to individual files in the output directory (default: ignore-<dataset_name>).
"""

import sys
import zarr
import numpy as np
import os
from PIL import Image
import argparse


class ZarrImageExtractor:
    def __init__(self, zarr_path, image_dataset_path, output_dir=None):
        self.zarr_path = zarr_path
        self.image_dataset_path = image_dataset_path
        
        # Set default output directory if not provided
        if output_dir is None:
            dataset_name = image_dataset_path.replace('/', '_').replace('\\', '_')
            self.output_dir = f"ignore-{dataset_name}"
        else:
            self.output_dir = output_dir
        
        # Load zarr store
        try:
            self.zarr_store = zarr.open(zarr_path, mode='r')
            print(f"Opened Zarr store: {zarr_path}")
            self.print_zarr_structure()
        except Exception as e:
            print(f"Error opening Zarr store: {e}")
            sys.exit(1)
        
        # Load image dataset
        try:
            self.images = self.zarr_store[image_dataset_path]
            print(f"Loaded image dataset: {image_dataset_path}")
            print(f"Shape: {self.images.shape}, dtype: {self.images.dtype}")
            # Convert shape to tuple to ensure we get an int
            shape = tuple(self.images.shape)
            self.num_images = shape[0]
            print(f"Total images: {self.num_images}")
        except KeyError:
            print(f"Error: Dataset '{image_dataset_path}' not found in Zarr store")
            print("Available datasets:")
            self.print_zarr_structure()
            sys.exit(1)
        except Exception as e:
            print(f"Error loading image dataset: {e}")
            sys.exit(1)
    
    def print_zarr_structure(self):
        """Print the structure of the Zarr store"""
        print("\nZarr Store Structure:")
        try:
            # Try to treat it as a group first
            try:
                for key in self.zarr_store.keys():  # type: ignore
                    dataset = self.zarr_store[key]
                    print(f"├── {key} {dataset.dtype} {dataset.shape}")
            except AttributeError:
                # Single array - doesn't have keys
                print(f"├── (root array) {self.zarr_store.dtype} {self.zarr_store.shape}")
        except Exception as e:
            print(f"Error reading Zarr structure: {e}")
    
    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Output directory: {self.output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            sys.exit(1)
    
    def normalize_image(self, img):
        """Normalize image data to 0-255 uint8 range"""
        if img.dtype == np.uint8:
            return img
        
        # Normalize to 0-1 range first
        img_min = np.min(img)
        img_max = np.max(img)
        
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = img
        
        # Convert to 0-255 uint8
        return (img_norm * 255).astype(np.uint8)
    
    def extract_images(self):
        """Extract all images to files"""
        self.create_output_directory()
        
        print(f"\nExtracting {self.num_images} images to {self.output_dir}/...")
        
        for i in range(self.num_images):
            try:
                # Get current image
                img = np.array(self.images[i])
                
                # Determine file format and process image
                if len(img.shape) == 3 and img.shape[-1] == 3:
                    # RGB image
                    img_norm = self.normalize_image(img)
                    pil_img = Image.fromarray(img_norm, 'RGB')
                    filename = f"image_{i:06d}.png"
                elif len(img.shape) == 2:
                    # Grayscale image
                    img_norm = self.normalize_image(img)
                    pil_img = Image.fromarray(img_norm, 'L')
                    filename = f"image_{i:06d}.png"
                elif len(img.shape) == 3 and img.shape[-1] == 4:
                    # RGBA image
                    img_norm = self.normalize_image(img)
                    pil_img = Image.fromarray(img_norm, 'RGBA')
                    filename = f"image_{i:06d}.png"
                else:
                    print(f"Warning: Unsupported image shape {img.shape} for image {i}, saving as numpy array")
                    filename = f"image_{i:06d}.npy"
                    filepath = os.path.join(self.output_dir, filename)
                    np.save(filepath, img)
                    continue
                
                # Save the image
                filepath = os.path.join(self.output_dir, filename)
                pil_img.save(filepath)
                
                # Progress indicator
                if (i + 1) % 100 == 0 or i == self.num_images - 1:
                    print(f"Extracted {i + 1}/{self.num_images} images")
                    
            except Exception as e:
                print(f"Error extracting image {i}: {e}")
                continue
        
        print(f"\nExtraction completed! Images saved to: {self.output_dir}")
        print(f"Total images extracted: {self.num_images}")


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python3 viz-image-to-file.py <zarr_path> <image_dataset_path> [output_dir]")
        print("\nExample:")
        print("  python3 viz-image-to-file.py 1.zarr data/left_wrist_img")
        print("  python3 viz-image-to-file.py 1.zarr data/left_wrist_img my_output_folder")
        print("\nThis will extract all images from the specified dataset in the Zarr store")
        print("to individual files in the output directory.")
        print("Default output directory: ignore-<dataset_name>")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    image_dataset_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) == 4 else None
    
    # Check if zarr path exists
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr store '{zarr_path}' not found")
        sys.exit(1)
    
    try:
        extractor = ZarrImageExtractor(zarr_path, image_dataset_path, output_dir)
        extractor.extract_images()
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()