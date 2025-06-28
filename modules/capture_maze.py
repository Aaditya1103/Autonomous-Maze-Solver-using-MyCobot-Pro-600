import cv2
import time

def crop_image(image, margins):
    """
    Crops an image by removing specified margins from each edge.

    :param image: Input image to crop.
    :param margins: Dictionary with keys 'top', 'bottom', 'left', 'right' specifying the margins.
    """
    height, width = image.shape[:2]
    start_x = margins['left']
    start_y = margins['top']
    end_x = width - margins['right']
    end_y = height - margins['bottom']

    if start_x >= end_x or start_y >= end_y:
        raise ValueError("Margins are too large, resulting in a negative or zero crop dimension.")

    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image

def main():
    try:
        cap = cv2.VideoCapture(0)  # 0 is usually the default ID for the built-in webcam
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        time.sleep(2)

        # Load the PNG image instead of using the webcam frame
        image_path = 'D:/Gracious/SEM-1/RAS-545/final project/version2/modules/captured_photo.jpg'  # Path to your PNG image 
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Failed to load image from {image_path}")

        # Define the margins for cropping
        margins = {
            'top': 100,    # Margin from the top edge
            'bottom': 100, # Margin from the bottom edge
            'left': 100,   # Margin from the left edge
            'right': 100   # Margin from the right edge
        }

        # # Capture the Image
        # ret, frame = cap.read()
        # if not ret:
        #     raise IOError("Failed to capture image from webcam")

        #  # Crop the image
        # cropped_image = crop_image(frame, margins)

        # Crop the image
        cropped_image = crop_image(image, margins)

        # Save the image
        output_path = 'cropped_maze.png'
        cv2.imwrite(output_path, cropped_image)
        print(f"Cropped image saved to {output_path}")
        
        cap.release()
        cv2.destroyAllWindows()
    except IOError as e:
        print(e)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
