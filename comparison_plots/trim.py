from PIL import Image
import os


def remove_white_and_crop(image_path, output_path):
    # Open the image
    img = Image.open(image_path).convert("RGBA")
    
    # Get the data of the image
    data = img.getdata()
    
    # Create a new list to store the modified pixels
    new_data = []
    
    for item in data:
        # Change all white (also shades of whites)
        # pixels to transparent
        if item[0] > 240 and item[1] > 240 and item[2] > 240:
            # Set transparency (R, G, B, A)
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    
    # Update image data
    img.putdata(new_data)
    
    # Crop the image to the bounding box of non-transparent pixels
    bbox = img.getbbox()
    if bbox:
        img_cropped = img.crop(bbox)
        # Save the image with transparency
        img_cropped.save(output_path, format="PNG", dpi=(400, 400))

# Define the paths to your images
input_images = ["init_graph/nba_angle.png", "init_graph/nba_error.png"]
output_images = ["init_graph/nba_angle_cropped.png", "init_graph/nba_error_cropped.png"]

# Remove white and crop for each image
for input_img, output_img in zip(input_images, output_images):
    remove_white_and_crop(input_img, output_img)
