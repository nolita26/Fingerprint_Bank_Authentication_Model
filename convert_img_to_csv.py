import os
from PIL import Image
import numpy as np
import csv
import pandas as pd

def convert_image_folder_to_csv(image_folder_path, csv_file_path):
  # Create a list of all the image files in the folder.
  image_files = os.listdir(image_folder_path)

  # Create a CSV writer object.
  with open(csv_file_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header row.
    csv_writer.writerow(["filename", "width", "height", "pixel_values"])

    # Iterate over the image files and write them to the CSV file.
    for image_file in image_files:
      if image_file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.BMP')):
        image_path = os.path.join(image_folder_path, image_file)
        # Open the image file.
        image = Image.open(image_path)

        # Get the image width and height.
        width, height = image.size

        # Get the pixel values of the image.
        pixel_values = np.array(image)

        # Write the image data to the CSV file.
        csv_writer.writerow([image_file, width, height, pixel_values])

# Example usage:
convert_image_folder_to_csv("/Users/Downloads/images", "/Users/Downloads/fingerprints.csv")