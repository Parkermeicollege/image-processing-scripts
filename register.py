import cv2
import os
import cv2
import numpy as np
import os
import skimage.io as io
import skimage.registration as registration
import skimage.transform as tf

def register_tiff_files(directory, reference_image_path):
    """Registers a directory of TIFF files using OpenCV's ECC algorithm.

    Args:
        directory: The directory containing the TIFF files to register.
        reference_image_path: The path to the reference image.
    """
    # Load the reference image
    reference_image = io.imread(reference_image_path, plugin='tifffile')
    reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Iterate through the directory and register each image
    for filename in os.listdir(directory):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            filepath = os.path.join(directory, filename)
            
            # Skip if it's the reference image
            if filepath == reference_image_path:
                continue
            
            # Load the image to register
            image_to_register = io.imread(filepath, plugin='tifffile')
            image_to_register_gray = cv2.cvtColor(image_to_register, cv2.COLOR_BGR2GRAY)

            # Define the motion model (here, we use affine transformation)
            warp_mode = cv2.MOTION_AFFINE

            # Define 2x3 or 3x3 matrices and initialize the matrix to identity
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                warp_matrix = np.eye(3, 3, dtype=np.float32)
            else:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            # Specify the number of iterations and termination criteria
            number_of_iterations = 50
            termination_eps = 1e-10

            # Define termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        number_of_iterations, termination_eps)

            # Perform the ECC alignment
            try:
                (cc, warp_matrix) = cv2.findTransformECC(reference_image_gray,
                                                        image_to_register_gray,
                                                        warp_matrix,
                                                        warp_mode,
                                                        criteria)
                
                # Apply the transformation to the image
                if warp_mode == cv2.MOTION_HOMOGRAPHY:
                    # Use warpPerspective for Homography 
                    aligned_image = cv2.warpPerspective(image_to_register,
                                                        warp_matrix,
                                                        (reference_image_gray.shape[1],
                                                        reference_image_gray.shape[0]),
                                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                else:
                    # Use warpAffine for Translation, Euclidean and Affine
                    aligned_image = cv2.warpAffine(image_to_register,
                                                    warp_matrix,
                                                    (reference_image_gray.shape[1],
                                                    reference_image_gray.shape[0]),
                                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

                # Save the registered image
                io.imsave(filepath, aligned_image, plugin='tifffile')
                print(f"Registered and saved: {filename}")

            except cv2.error as e:
                print(f"Error registering {filename}: {e}")

# Example usage:
image_directory = "/Users/parkermei/Projects/ISAM/Project 4/REG_IMGS"  # Replace with your directory
reference_image_path = "/Users/parkermei/Projects/ISAM/Project 4/REG_IMGS/DSC_0031.tiff"  # Replace with your reference image path
register_tiff_files(image_directory, reference_image_path)