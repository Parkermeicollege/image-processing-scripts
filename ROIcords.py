import cv2
import os
import rawpy

# Set the path to the .NEF image file
image_path = "/Users/parkermei/Projects/ISAM/Project 4/ISO800/DSC_0040.NEF"

def select_bounding_box(image_path):
    raw = rawpy.imread(image_path)
    rgb = raw.postprocess(use_camera_wb=True)
    roi = cv2.selectROI("Select ROI", rgb, False, True)
    cv2.destroyAllWindows()
    return roi

ROI = select_bounding_box(image_path)
print(ROI)