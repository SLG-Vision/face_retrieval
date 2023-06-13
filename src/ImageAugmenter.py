import cv2
import random
import numpy as np

class ImageAugmenter:
    def __init__(self, usingSuggestedTransforms=True, usingAllTransforms=False, transformList=[]):
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
        
        if(sum([usingSuggestedTransforms, usingAllTransforms, len(transformList) > 0]) > 1):
            raise Exception("Only one of the 2 parameters can be set to True and/or the transformList must be empty.")
        
        self._transforms = []

        if(len(transformList) > 0):
            self._transforms = transformList
        
        
        if(usingSuggestedTransforms):
            self._transforms = ["resize", "blur", "motion_blur", "rotate", "brightness", "contrast", "saturation", "tilt", "translate"]
            
        if(usingAllTransforms):
            for key in self.available_transforms.keys():
                self._transforms.append(key)
        

    def apply_transforms(self, image):
        
        for transform_name in self._transforms:
            if transform_name in self.available_transforms:
                transform_func = self.available_transforms[transform_name]
                image = transform_func(image)
            else:
                print(f"Transform '{transform_name}' not available.")
        return image

    def resize(self, image):
        rows, cols = image.shape[:2]
        width = random.randint(int(0.8 * cols), int(1.2 * cols))
        height = random.randint(int(0.8 * rows), int(1.2 * rows))
        resized_image = cv2.resize(image, (width, height))
        return resized_image

    def blur(self, image):
        kernel_size = random.randint(3, 9)
        if kernel_size % 2 == 0:    # il kernel deve essere di dim. dispari
            kernel_size += 1
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred_image

    def motion_blur(self, image):
        kernel_size = random.randint(3, 9)
        kernel = self.generate_motion_blur_kernel(kernel_size)
        motion_blur = cv2.filter2D(image, -1, kernel)
        return motion_blur

    def rotate(self, image):
        angle = random.randint(-30, 30)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        return rotated_image

    def flip(self, image):
        flip_code = random.randint(-1, 1)
        flipped_image = cv2.flip(image, flip_code)
        return flipped_image

    def adjust_brightness(self, image):
        factor = random.uniform(0.7, 1.4)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.multiply(v, factor)
        v = np.clip(v, 0, 255)
        adjusted_image = cv2.merge((h, s, v))
        adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_HSV2BGR)
        return adjusted_image

    def adjust_contrast(self, image):
        factor = random.uniform(0.7, 1.4)
        adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted_image

    def adjust_saturation(self, image):
        factor = random.uniform(0.7, 2.0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * factor, 0, 255).astype(np.uint8)
        adjusted_image = cv2.merge((h, s, v))
        adjusted_image = cv2.cvtColor(adjusted_image, cv2.COLOR_HSV2BGR)
        return adjusted_image


    def zoom(self, image):
        scale = random.uniform(0.1, 0.3)
        rows, cols = image.shape[:2]
        cx = int(cols / 2)
        cy = int(rows / 2)
        new_width = int(cols * scale)
        new_height = int(rows * scale)
        x = cx - new_width // 2
        y = cy - new_height // 2
        zoomed_image = image[y:y+new_height, x:x+new_width]
        zoomed_image = cv2.resize(zoomed_image, (cols, rows))
        return zoomed_image

    def tilt(self, image):
        angle = random.randint(-10, 10)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        tilted_image = cv2.warpAffine(image, M, (cols, rows))
        return tilted_image

    def translate(self, image):
        shift_x = random.randint(-50, 50)
        shift_y = random.randint(-50, 50)
        rows, cols = image.shape[:2]
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])  # type: ignore
        translated_image = cv2.warpAffine(image, M, (cols, rows))
        return translated_image

    def generate_motion_blur_kernel(self, kernel_size):
        kernel = np.zeros((kernel_size, kernel_size))
        rand_angle = random.uniform(0, 180)
        center = kernel_size // 2
        cv2.ellipse(kernel, (center, center), (center, 0), rand_angle, 0, 360, 255, -1)
        kernel = kernel / np.sum(kernel)
        return kernel



# most aggressive transforms: flip, zoom

augmenter = ImageAugmenter()

vid = cv2.VideoCapture(0)

while(True):
    _, frame = vid.read()
    cv2.imshow('Original Image', frame)
    augmented_image = augmenter.apply_transforms(frame)
    cv2.imshow("Augmented Image", augmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
