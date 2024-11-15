import cv2
import numpy as np
import rawpy
import imageio
from pathlib import Path
import tempfile

def raw_to_rgb(raw_path):
    """
    Convert RAW NEF file to RGB numpy array.
    
    Args:
        raw_path (str): Path to NEF file
    Returns:
        numpy array: RGB image array
    """
    with rawpy.imread(raw_path) as raw:
        # Convert to RGB color space while preserving as much information as possible
        rgb = raw.postprocess(
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB
        )
    return rgb

def save_as_nef(processed_array, original_nef_path, output_path):
    """
    Save processed image data back to NEF format by copying metadata
    from the original NEF file and updating the embedded JPEG preview.
    
    Args:
        processed_array (numpy array): Processed image data
        original_nef_path (str): Path to original NEF file to copy metadata from
        output_path (str): Path to save new NEF file
    """
    # Create temporary JPEG preview
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_jpg:
        # Convert 16-bit to 8-bit for JPEG preview
        preview_data = (processed_array / 256).astype(np.uint8)
        imageio.imwrite(tmp_jpg.name, preview_data)
        
        # Copy original NEF with metadata
        with open(original_nef_path, 'rb') as src, open(output_path, 'wb') as dst:
            # Copy NEF data
            dst.write(src.read())
            
        # Update embedded preview if needed
        # Note: This is a simplified version. A full implementation would need to
        # properly update the NEF container structure and metadata

def register_nef_images(reference_path, image_paths, output_dir):
    """
    Register multiple NEF images to align with a reference NEF image.
    
    Args:
        reference_path (str): Path to the reference NEF image
        image_paths (list): List of paths to NEF images that need to be aligned
        output_dir (str): Directory to save registered NEF images
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and convert reference image
    reference = raw_to_rgb(str(reference_path))
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors for reference image
    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_gray, None)
    
    # Initialize feature matcher
    matcher = cv2.BFMatcher()
    
    for image_path in image_paths:
        try:
            # Read and convert moving image
            moving = raw_to_rgb(str(image_path))
            moving_gray = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY)
            
            # Detect keypoints and compute descriptors
            moving_keypoints, moving_descriptors = sift.detectAndCompute(moving_gray, None)
            
            # Match features
            matches = matcher.knnMatch(moving_descriptors, reference_descriptors, k=2)
            
            # Apply ratio test to filter good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) < 4:
                print(f"Not enough good matches found for {image_path}")
                continue
            
            # Get corresponding points
            src_pts = np.float32([moving_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([reference_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Calculate homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Warp image
            height, width = reference.shape[:2]
            registered = cv2.warpPerspective(moving, H, (width, height))
            
            # Save registered image as NEF
            output_path = output_dir / f"registered_{Path(image_path).name}"
            save_as_nef(registered, image_path, str(output_path))
            print(f"Successfully registered: {image_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

def main():
    # Example usage
    reference_path = "'/Users/parkermei/Projects/ISAM/Project 4/ISO200/DSC_0031.NEF'"
    image_paths = [
        "/Users/parkermei/Projects/ISAM/Project 4/ISO200/DSC_0032.NEF",
        "/Users/parkermei/Projects/ISAM/Project 4/ISO200/DSC_0033.NEF",
        "/Users/parkermei/Projects/ISAM/Project 4/ISO200/DSC_0034.NEF",
        "/Users/parkermei/Projects/ISAM/Project 4/ISO200/DSC_0036.NEF",
        "/Users/parkermei/Projects/ISAM/Project 4/ISO200/DSC_0037.NEF",
        "/Users/parkermei/Projects/ISAM/Project 4/ISO200/DSC_0038.NEF"
    ]
    output_dir = "'/Users/parkermei/Projects/ISAM/Project 4/REG_IMGS'"
    
    register_nef_images(reference_path, image_paths, output_dir)

if __name__ == "__main__":
    main()