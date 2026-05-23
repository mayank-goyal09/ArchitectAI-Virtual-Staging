import cv2
import numpy as np

class RoomProcessor:
    def __init__(self, low_threshold=50, high_threshold=150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def extract_structure(self, image_path):
        """
        Turns a room photo into a structural line map.
        """
        # 1. Load image
        img = cv2.imread(image_path)
        if img is None:
            return "❌ Error: Image not found!"

        # 2. Convert to Grayscale (AI doesn't need color for lines)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # 3. Reduce noise (Smooth out the wall textures)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4. Canny Edge Detection (The Magic Step)
        # It finds where the brightness changes suddenly (corners/edges)
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)

        # 5. Dilate lines slightly (Makes it easier for ControlNet to follow)
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        return dilated_edges

    def save_map(self, processed_img, output_path):
        cv2.imwrite(output_path, processed_img)
        print(f"✅ Skeleton map saved to: {output_path}")