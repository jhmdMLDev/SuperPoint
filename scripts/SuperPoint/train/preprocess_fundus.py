import os

import numpy as np
import cv2


def crop_circle(image):
    # Convert the image to binary
    _, binary_image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:
        # Find the contour with the largest area (assuming it's the circular shape)
        max_contour = max(contours, key=cv2.contourArea)

        # Get the center and radius of the bounding circle
        moments = cv2.moments(max_contour)
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )

        # Calculate the radius based on the area of the contour
        radius = int(np.sqrt(cv2.contourArea(max_contour) / np.pi))

        # Calculate the bounding box around the circle
        box_size = int(radius / np.sqrt(2))
        start_row = max(0, center[1] - box_size)
        end_row = min(image.shape[0], center[1] + box_size)
        start_col = max(0, center[0] - box_size)
        end_col = min(image.shape[1], center[0] + box_size)

        # Crop the circular region
        cropped_img = image[start_row:end_row, start_col:end_col]
    else:
        height, width = image.shape[:2]
        start_row, start_col = int(height * 0.1), int(width * 0.15)
        end_row, end_col = int(height * 0.9), int(width * 0.85)
        cropped_img = image[start_row:end_row, start_col:end_col]
    return cropped_img


def crop_and_resize_images_recursive(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Loop through the source directory and its subdirectories recursively
    file_num = 0
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            # Check if the file is a JPG or PNG image
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(root, filename)

                # Read the image in grayscale
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                cropped_img = crop_circle(img)

                # Resize the image to 512x512
                resized_img = cv2.resize(cropped_img, (512, 512))

                # Generate a unique name for the saved image
                output_filename = f"cropped_resized_{file_num}.png"
                file_num += 1

                # Save the image to the destination directory
                output_filepath = os.path.join(destination_dir, output_filename)
                cv2.imwrite(output_filepath, resized_img)

                print(f"Processed: {filename} -> {output_filename}")


# Example usage:
source_directory = r"C:\Users\Javad\Downloads\FundusImageTest"
destination_directory = (
    r"C:\Users\Javad\Desktop\Dataset\SuperpointDataset\FundusAugmentationVal"
)
crop_and_resize_images_recursive(source_directory, destination_directory)
