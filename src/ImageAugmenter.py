import numpy as np
import random
import cv2

class ImageAugmenter:
    def __init__(self):
        self.available_transforms = {
            "resize": self.resize,
            "blur": self.blur,
            "motion_blur": self.motion_blur,
            "rotate": self.rotate,
            "flip": self.flip,
            "brightness": self.adjust_brightness,
            "contrast": self.adjust_contrast,
            "saturation": self.adjust_saturation,
            "zoom": self.zoom,
            "tilt": self.tilt,
            "translate": self.translate
        }

    def apply_transforms(self, image, transforms_list):
        for transform_name in transforms_list:
            if transform_name in self.available_transforms:
                transform_func = self.available_transforms[transform_name]
                image = transform_func(image)
            else:
                print(f"Transform '{transform_name}' not available.")
        return image

    def resize(self, image, width, height):
        resized_image = cv2.resize(image, (width, height))
        return resized_image

    def blur(self, image, kernel_size):
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image

    def motion_blur(self, image, kernel_size):
        kernel = self.generate_motion_blur_kernel(kernel_size)
        motion_blur = cv2.filter2D(image, -1, kernel)
        return motion_blur

    def rotate(self, image, angle):
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        return rotated_image

    def flip(self, image):
        flipped_image = cv2.flip(image, 1)
        return flipped_image

    def adjust_brightness(self, image, factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, factor)
        v = np.clip(v, 0, 255)
        adjusted_image = cv2.merge((h, s, v))
        adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_HSV2BGR)
        return adjusted_image

    def adjust_contrast(self, image, factor):
        adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted_image

    def adjust_saturation(self, image, factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, factor)
        s = np.clip(s, 0, 255)
        adjusted_image = cv2.merge((h, s, v))
        adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_HSV2BGR)
        return adjusted_image

    def zoom(self, image, zoom_factor):
        rows, cols = image.shape[:2]
        cx = int(cols / 2)
        cy = int(rows / 2)
        new_width = int(cols * zoom_factor)
        new_height = int(rows * zoom_factor)
        x = cx - new_width // 2
        y = cy - new_height // 2
        zoomed_image = cv2.resize(image[y:y+new_height, x:x+new_width], (cols, rows))
        return zoomed_image

    def tilt(self, image, angle):
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        tilted_image = cv2.warpAffine(image, M, (cols, rows))
        return tilted_image

    def translate(self, image, shift_x, shift_y):
        rows, cols = image.shape[:2]
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]]) # type: ignore
        translated_image = cv2.warpAffine(image, M, (cols, rows))
        return translated_image

    def generate_motion_blur_kernel(self, kernel_size):
        kernel = np.zeros((kernel_size, kernel_size))
        rand_angle = random.uniform(0, 180)
        center = kernel_size // 2
        cv2.ellipse(kernel, (center, center), (center, 0), rand_angle, 0, 360, 255, -1)
        kernel = kernel / np.sum(kernel)
        return kernel



# test code

image = cv2.imread("path_to_image.jpg") 

augmenter = ImageAugmenter()
transforms_list = ["resize", "blur", "zoom", "tilt"] 
augmented_image = augmenter.apply_transforms(image, transforms_list)
cv2.imshow("Augmented Image", augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
