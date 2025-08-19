#!/usr/bin/env python3
"""
Zarr Image Visualizer

Usage:
    python3 viz-image.py <zarr_path> <image_dataset_path>

Example:
    python3 viz-image.py 1.zarr data/left_wrist_img

Navigate through images:
    - Press SPACE to go to next image
    - Press 'q' or ESC to quit
    - Press 'b' to go back to previous image
"""

import sys
import cv2
import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os


class ZarrImageVisualizer:
    def __init__(self, zarr_path, image_dataset_path):
        self.zarr_path = zarr_path
        self.image_dataset_path = image_dataset_path
        self.current_index = 0

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

        # Setup matplotlib
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Add navigation buttons
        self.setup_buttons()

        # Display first image
        self.update_display()

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

    def setup_buttons(self):
        """Setup navigation buttons"""
        # Previous button
        ax_prev = plt.axes((0.1, 0.02, 0.1, 0.04))
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_image)

        # Next button
        ax_next = plt.axes((0.8, 0.02, 0.1, 0.04))
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_image)

        # Info text
        ax_info = plt.axes((0.25, 0.02, 0.5, 0.04))
        ax_info.text(0.5, 0.5, 'SPACE: Next | B: Previous | Q/ESC: Quit',
                    ha='center', va='center', transform=ax_info.transAxes,
                    fontsize=10)
        ax_info.set_xticks([])
        ax_info.set_yticks([])

    def update_display(self):
        """Update the displayed image"""
        if self.num_images == 0:
            print("No images to display")
            return

        # Ensure index is within bounds
        self.current_index = max(0, min(self.current_index, self.num_images - 1))

        # Get current image
        try:
            # Convert Zarr array slice to numpy array
            img = np.array(self.images[self.current_index])

            # Clear the axis
            self.ax.clear()

            # Display the image
            if len(img.shape) == 3 and img.shape[-1] == 3:
                # RGB image
                if img.dtype == np.uint8:
                    self.ax.imshow(img)
                    cv2.imwrite(f"image_{self.current_index}.png", img)
                else:
                    # Normalize if not uint8
                    img_min = np.min(img)
                    img_max = np.max(img)
                    if img_max > img_min:
                        img_norm = (img - img_min) / (img_max - img_min)
                    else:
                        img_norm = img
                    self.ax.imshow(img_norm)
            elif len(img.shape) == 2:
                # Grayscale image
                self.ax.imshow(img, cmap='gray')
            else:
                print(f"Unsupported image shape: {img.shape}")
                return

            # Set title with image info
            img_min = np.min(img)
            img_max = np.max(img)
            self.ax.set_title(f"Image {self.current_index + 1}/{self.num_images}\n"
                            f"Shape: {img.shape}, dtype: {img.dtype}\n"
                            f"Min: {img_min:.2f}, Max: {img_max:.2f}")

            # Remove axis ticks for cleaner display
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            # Refresh the display
            self.fig.canvas.draw()

            print(f"Displaying image {self.current_index + 1}/{self.num_images}")

        except Exception as e:
            print(f"Error displaying image {self.current_index}: {e}")

    def next_image(self, event=None):
        """Move to next image"""
        if self.current_index < self.num_images - 1:
            self.current_index += 1
            self.update_display()
        else:
            print("Already at the last image")

    def prev_image(self, event=None):
        """Move to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
        else:
            print("Already at the first image")

    def on_key_press(self, event):
        """Handle key press events"""
        if event.key == ' ':  # Space key
            self.next_image()
        elif event.key == 'b':  # B key
            self.prev_image()
        elif event.key in ['q', 'escape']:  # Quit
            plt.close()
        elif event.key == 'right':  # Arrow keys
            self.next_image()
        elif event.key == 'left':
            self.prev_image()

    def show(self):
        """Show the visualization"""
        plt.tight_layout()
        plt.show()


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 viz-image.py <zarr_path> <image_dataset_path>")
        print("\nExample:")
        print("  python3 viz-image.py 1.zarr data/left_wrist_img")
        print("\nThis will visualize images from the specified dataset in the Zarr store.")
        print("Use SPACE to navigate to the next image, 'b' for previous, 'q' to quit.")
        sys.exit(1)

    zarr_path = sys.argv[1]
    image_dataset_path = sys.argv[2]

    # Check if zarr path exists
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr store '{zarr_path}' not found")
        sys.exit(1)

    try:
        visualizer = ZarrImageVisualizer(zarr_path, image_dataset_path)
        visualizer.show()
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
