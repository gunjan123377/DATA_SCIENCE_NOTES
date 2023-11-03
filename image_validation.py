import cv2
import numpy as np

def match_template(main_image_path, template_image_path, threshold=0.8):
    # Load the main image (the one you want to search within)
    main_image = cv2.imread(main_image_path)
    # Load the template image (the part you want to find)
    template_image = cv2.imread(template_image_path)

    # Convert both images to grayscale
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find the maximum matching value and its location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the maximum matching value is above the threshold
    if max_val >= threshold:
        return True
    else:
        return False

# Example usage
if match_template('main_image.png', 'template_image.png', threshold=0.8):
    print("Match found (Above 80%)")
else:
    print("Match not found (Below 80%)")
