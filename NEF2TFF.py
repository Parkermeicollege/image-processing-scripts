import rawpy
import os
from PIL import Image

def convert_nef_to_tiff(input_dir, output_dir):
  """
  Converts NEF images in the input directory to TIFF format and saves them in the output directory.

  Args:
    input_dir: The path to the input directory containing NEF images.
    output_dir: The path to the output directory for TIFF images.
  """

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for filename in os.listdir(input_dir):
    if filename.endswith('.NEF'):
      input_path = os.path.join(input_dir, filename)
      output_path = os.path.join(output_dir, filename[:-4] + '.tiff')

      with rawpy.imread(input_path) as raw:
        rgb = raw.postprocess()
        raw.close()

        tiff_image = Image.fromarray(rgb)
        tiff_image.save(output_path)

# Example usage:
input_directory = '/Users/parkermei/Projects/ISAM/Project 4/ISO200'
output_directory = '/Users/parkermei/Projects/ISAM/Project 4/REG_IMGS'

convert_nef_to_tiff(input_directory, output_directory)