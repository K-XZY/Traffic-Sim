from PIL import Image
import numpy as np

def process_image(input_path, output_path, threshold=128):
    """
    Load an image, invert its colors, and then apply thresholding to push colors to black or white.
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path where the processed image will be saved
        threshold (int): Value between 0-255 to determine black/white cutoff (default: 128)
    """
    try:
        # Open the image
        img = Image.open(input_path)
        
        # Convert image to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array for easier manipulation
        img_array = np.array(img)
        
        # Step 1: Invert the colors (255 - original value)
        inverted_array = 255 - img_array
        
        # Step 2: Convert to grayscale using standard luminance formula
        # Using weights: R: 0.299, G: 0.587, B: 0.114
        grayscale = np.dot(inverted_array[..., :3], [0.299, 0.587, 0.114])
        
        # Step 3: Apply thresholding
        binary = np.where(grayscale > threshold, 255, 0)
        
        # Convert back to RGB format
        binary_rgb = np.stack([binary] * 3, axis=-1)
        
        # Convert back to PIL Image
        final_image = Image.fromarray(binary_rgb.astype(np.uint8))
        
        # Save the processed image
        final_image.save(output_path)
        print(f"Successfully processed image and saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Example usage
if __name__ == "__main__":
    input_path = "temp.jpg"  # Replace with your input image path
    output_path = "temp_inverted.jpg"  # Replace with desired output path
    process_image(input_path, output_path)
