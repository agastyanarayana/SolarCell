import cv2
import numpy as np
import os

def halfcrop(folder_path):
    # Check if the folder path exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder path '{folder_path}' does not exist.")

    # Create the 'halfcrop' folder inside the given folder
    halfcrop_folder = os.path.join(folder_path, 'halfcrop')
    os.makedirs(halfcrop_folder, exist_ok=True)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for idx, image_file in enumerate(image_files):
        # Construct the full path to the image file
        image_path = os.path.join(folder_path, image_file)

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping {image_file}. Failed to load image.")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate column-wise sums
        column_sums = np.sum(gray, axis=0)
        row_sums = np.sum(gray, axis=1)
        
        # Find the longest subarray where all values are less than 1000
        max_length = 0
        start_index = -1
        end_index = -1
        current_length = 0
        current_start = -1

        for i in range(len(column_sums)):
            if column_sums[i] < 1000:
                if current_start == -1:
                    current_start = i
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    start_index = current_start
                    end_index = i - 1
                current_length = 0
                current_start = -1

        if current_length > max_length:
            max_length = current_length
            start_index = current_start
            end_index = len(column_sums) - 1

        max_length2 = 0
        start_index2 = -1
        end_index2 = -1
        current_length2 = 0
        current_start2 = -1

        for i in range(len(row_sums)):
            if row_sums[i] < 500:
                if current_start2 == -1:
                    current_start2 = i
                current_length2 += 1
            else:
                if current_length2 > max_length2:
                    max_length2 = current_length2
                    start_index2 = current_start2
                    end_index2 = i - 1
                current_length2 = 0
                current_start2 = -1

        if current_length2 > max_length2:
            max_length2 = current_length2
            start_index2 = current_start2
            end_index2 = len(row_sums) - 1

        cropped_left_image = image[:start_index2, :start_index]
        cropped_right_image = image[:start_index2, end_index+1:]

        # Create a folder for the current image inside the 'halfcrop' folder
        image_folder = os.path.join(halfcrop_folder, image_file.split('.')[0])
        os.makedirs(image_folder, exist_ok=True)

        # Save the cropped images with distinct filenames
        output_left_path = os.path.join(image_folder, f"L_{image_file}")
        output_right_path = os.path.join(image_folder, f"R_{image_file}")

        cv2.imwrite(output_left_path, cropped_left_image)
        cv2.imwrite(output_right_path, cropped_right_image)

        print(f"Processed {image_file}. Cropped images saved as {output_left_path} and {output_right_path}")

    print("Processing complete.")

# Example usage:
#folder_path = '/content/drive/MyDrive/Trial'  # Replace with the path to your folder containing images
#halfcrop(folder_path)
