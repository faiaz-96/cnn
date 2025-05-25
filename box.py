import cv2

import cv2

def draw_bbox(image_path, bbox, output_path="output.jpg", color=(255, 0, 0), thickness=2):
    """
    Draws a bounding box on the image using normalized coordinates (0–1).

    Parameters:
    - image_path: str, path to the input image
    - bbox: tuple, (x_center, y_center, width, height) in normalized units (0–1)
    - output_path: str, path to save the output image
    - color: tuple, BGR color for the box (default: blue)
    - thickness: int, line thickness of the box
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    h, w = image.shape[:2]

    # Unpack and scale bbox
    x_center, y_center, box_width, box_height = bbox
    x_center *= w
    y_center *= h
    box_width *= w
    box_height *= h

    # Convert to top-left and bottom-right corners
    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Save the result
    cv2.imwrite(output_path, image)
    print(f"Saved output to {output_path}")

draw_bbox('handw_s_demo/1.jpg', (0.290348, 0.24344, 0.203471, 0.231612))


