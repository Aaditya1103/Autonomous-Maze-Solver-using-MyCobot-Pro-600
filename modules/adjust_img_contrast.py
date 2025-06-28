import cv2
import numpy as np

def convert_to_min_rgb_grayscale(image_path):
    """
    Converts an image to grayscale by taking the minimum value of the RGB channels for each pixel.
    
    :param image_path: Path to the image file.
    :return: Grayscale image where each pixel is the minimum value of its RGB components.
    """
    # Load the image in color
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Could not load image.")
        return None

    # Calculate the minimum value across the RGB channels for each pixel
    min_rgb_image = np.min(image, axis=2)

    return min_rgb_image

def apply_threshold(image, threshold=128):
    """
    Converts an image to black and white based on a given threshold.

    :param image_path: Path to the input image.
    :param threshold: Pixel intensity threshold (0-255).
    """
    # Load the image in grayscale mode
    if image is None:
        raise FileNotFoundError("The specified image file was not found.")

    # Apply thresholding
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return binary_image

def save_image(image, output_path):
    """
    Saves the processed image to a file.

    :param image: The image to save.
    :param output_path: Path where the image will be saved.
    """
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

def main():
    image_path = 'cropped_maze.png'
    #image_path = 'test_img.png'
    output_path = 'bw_maze_cropped.png'

    try:
        grayscale_image = convert_to_min_rgb_grayscale(image_path)
        # Process the image
        binary_image = apply_threshold(grayscale_image, threshold=120)
        
        # Save the thresholded image
        save_image(binary_image, output_path)

        # Optionally display the image
    #    cv2.imshow("Thresholded Image", binary_image)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
