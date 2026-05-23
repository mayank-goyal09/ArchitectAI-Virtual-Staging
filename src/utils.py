from PIL import Image

def resize_image(image, size=(512, 512)):
    """
    Helper to resize images for the model.
    """
    return image.resize(size)
