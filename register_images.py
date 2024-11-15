import os
import skimage.transform as tf
import skimage.io as io

def register_images(image1, image2):
    """Registers two images using phase correlation.

    Args:
        image1: The first image.
        image2: The second image.

    Returns:
        The registered second image.
    """

    shifts, error, diffphase = tf.register_translation(image1, image2)
    image2_shifted = tf.warp(image2, tf.AffineTransform(translation=shifts))
    return image2_shifted

def register_directory(input_dir, output_dir):
    """Registers images in a directory.

    Args:
        input_dir: The input directory.
        output_dir: The output directory.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = sorted(os.listdir(input_dir))

    for i in range(len(image_files) - 1):
        image1_path = os.path.join(input_dir, image_files[i])
        image2_path = os.path.join(input_dir, image_files[i+1])

        image1 = io.imread(image1_path)
        image2 = io.imread(image2_path)

        registered_image = register_images(image1, image2)

        output_path = os.path.join(output_dir, f"registered_{image_files[i+1]}")
        io.imsave(output_path, registered_image)

# Example usage:
input_dir = "/Users/parkermei/Projects/ISAM/Project 4/REG_IMGS"
output_dir = "/Users/parkermei/Projects/ISAM/Project 4/Final_Imgs"

register_directory(input_dir, output_dir)