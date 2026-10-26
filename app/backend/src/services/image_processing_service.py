"""
Image Processing Service for Augmentation and Degradation
Adapted from src/augmentation.py
"""
import io
import random
from functools import lru_cache
from typing import Optional, Dict, List, Tuple, Callable

import cv2
import numpy as np
from PIL import Image, ImageFile
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Augmentation Constants
STRETCH_FACTOR_RANGE = (0.8, 1.2)
ROTATION_ANGLE_RANGE = (-15, 15)
ZOOM_FACTOR_RANGE = (1.0, 1.4)
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)
SATURATION_RANGE = (0.6, 1.4)
HUE_SHIFT_RANGE = (-20, 20)

# Degradation Probabilities and Ranges
DEGRADATION_PROBS = {
    'gaussian_noise': 0.3,
    'salt_pepper_noise': 0.2,
    'blur_gaussian': 0.4,
    'blur_motion': 0.3,
    'occlusion': 0.3,
    'random_masks': 0.3,
    'scratches': 0.25,
    'rain_drops': 0.4,
}

NOISE_VAR_RANGE = (0.005, 0.02)
SALT_PEPPER_AMOUNT_RANGE = (0.01, 0.05)
BLUR_KERNEL_RANGE = (3, 7)
MOTION_BLUR_KERNEL_RANGE = (5, 15)
OCCLUSION_PATCHES_RANGE = (2, 5)
RANDOM_MASKS_RANGE = (2, 5)
SCRATCHES_RANGE = (3, 8)
RAIN_DROPS_RANGE = (10, 30)


@lru_cache(maxsize=32)
def _motion_blur_kernel(kernel_size: int) -> np.ndarray:
    """Create motion blur kernel (cached for performance)"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = 1.0
    kernel /= kernel_size
    return kernel


@lru_cache(maxsize=8)
def _coordinate_grids(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create coordinate grids for distortion (cached for performance)"""
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    return grid_y, grid_x


class GroundTruthAugmentor:
    """
    Ground truth augmentation class for geometric and colorimetric transformations.
    Creates augmented clean images without degradation or black gaps.
    """

    def apply_stretch(
        self,
        image: np.ndarray,
        stretch_factor_x: Optional[float] = None,
        stretch_factor_y: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply stretch transformation without black gaps.
        Uses interpolation and crops back to original size.
        """
        if stretch_factor_x is None:
            stretch_factor_x = random.uniform(*STRETCH_FACTOR_RANGE)
        if stretch_factor_y is None:
            stretch_factor_y = random.uniform(*STRETCH_FACTOR_RANGE)

        h, w = image.shape[:2]
        new_w = int(w * stretch_factor_x)
        new_h = int(h * stretch_factor_y)

        stretched = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Crop or resize to maintain original size
        if new_w >= w and new_h >= h:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            result = stretched[start_y:start_y + h, start_x:start_x + w]
        else:
            result = cv2.resize(stretched, (w, h), interpolation=cv2.INTER_LINEAR)

        return result

    def apply_rotation(
        self,
        image: np.ndarray,
        angle: Optional[float] = None
    ) -> np.ndarray:
        """Apply rotation without black gaps using border reflection."""
        if angle is None:
            angle = random.uniform(*ROTATION_ANGLE_RANGE)

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)

        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]

        rotated = cv2.warpAffine(
            image, rotation_matrix, (bound_w, bound_h),
            borderMode=cv2.BORDER_REFLECT
        )

        crop_x = (bound_w - w) // 2
        crop_y = (bound_h - h) // 2
        cropped = rotated[crop_y:crop_y + h, crop_x:crop_x + w]

        return cropped

    def apply_zoom(
        self,
        image: np.ndarray,
        zoom_factor: Optional[float] = None
    ) -> np.ndarray:
        """Apply zoom (crop-based) without black gaps."""
        if zoom_factor is None:
            zoom_factor = random.uniform(*ZOOM_FACTOR_RANGE)

        h, w = image.shape[:2]
        new_h = int(h / zoom_factor)
        new_w = int(w / zoom_factor)

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        cropped = image[top:top + new_h, left:left + new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return zoomed

    def alter_colorimetry(
        self,
        image: np.ndarray,
        brightness_factor: Optional[float] = None,
        contrast_factor: Optional[float] = None,
        saturation_factor: Optional[float] = None,
        hue_shift: Optional[int] = None
    ) -> np.ndarray:
        """Modify brightness, contrast, saturation, and hue."""
        img_array = image.astype(np.float32)

        if brightness_factor is None:
            brightness_factor = random.uniform(*BRIGHTNESS_RANGE)
        img_array = img_array * brightness_factor

        if contrast_factor is None:
            contrast_factor = random.uniform(*CONTRAST_RANGE)
        mean = np.mean(img_array, axis=(0, 1), keepdims=True)
        img_array = (img_array - mean) * contrast_factor + mean

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

        if saturation_factor is None:
            saturation_factor = random.uniform(*SATURATION_RANGE)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor

        if hue_shift is None:
            hue_shift = random.randint(*HUE_SHIFT_RANGE)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img_array

    def apply_augmentation(
        self,
        image: np.ndarray,
        apply_stretch: bool = True,
        apply_rotation: bool = True,
        apply_zoom: bool = True,
        apply_color: bool = True
    ) -> np.ndarray:
        """Apply random ground truth augmentations (geometric + colorimetric)."""
        augmented = image.copy()
        augmentation_applied = False

        augmentations = [
            (apply_stretch, 0.4, self.apply_stretch),
            (apply_rotation, 0.4, self.apply_rotation),
            (apply_zoom, 0.4, self.apply_zoom),
            (apply_color, 0.5, self.alter_colorimetry),
        ]

        for should_apply, probability, func in augmentations:
            if should_apply and random.random() < probability:
                augmented = func(augmented)
                augmentation_applied = True

        # Ensure at least one augmentation is applied
        if not augmentation_applied:
            augmented = self.apply_rotation(augmented)

        return augmented


class ImageDegradator:
    """
    Image degradation class for applying realistic degradations to images.
    Used for creating degraded versions of clean images.
    """

    def add_gaussian_noise(
        self,
        image: np.ndarray,
        mean: float = 0,
        var: float = 0.01
    ) -> np.ndarray:
        """Add Gaussian (normal distribution) noise to image"""
        img_float = image.astype(np.float32) / 255.0
        noise = np.random.normal(mean, var**0.5, img_float.shape)
        noisy = np.clip(img_float + noise, 0, 1)
        return (noisy * 255).astype(np.uint8)

    def add_salt_pepper_noise(
        self,
        image: np.ndarray,
        amount: float = 0.05
    ) -> np.ndarray:
        """Add salt and pepper noise (random white and black pixels)"""
        noisy = image.copy()
        h, w = image.shape[:2]

        num_salt = int(np.ceil(amount * image.size * 0.5))
        coords_salt = [np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)]
        if image.ndim == 3:
            noisy[coords_salt[0], coords_salt[1], :] = 255
        else:
            noisy[coords_salt[0], coords_salt[1]] = 255

        num_pepper = int(np.ceil(amount * image.size * 0.5))
        coords_pepper = [np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper)]
        if image.ndim == 3:
            noisy[coords_pepper[0], coords_pepper[1], :] = 0
        else:
            noisy[coords_pepper[0], coords_pepper[1]] = 0

        return noisy

    def apply_blur(
        self,
        image: np.ndarray,
        blur_type: str = 'gaussian',
        kernel_size: int = 5
    ) -> np.ndarray:
        """Apply various types of blur"""
        if blur_type == 'gaussian':
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'motion':
            kernel = _motion_blur_kernel(kernel_size)
            blurred = cv2.filter2D(image, -1, kernel)
        elif blur_type == 'box':
            blurred = cv2.blur(image, (kernel_size, kernel_size))
        else:
            raise ValueError(f"Unknown blur type: {blur_type}")

        return blurred

    def add_occlusion(
        self,
        image: np.ndarray,
        num_patches: int = 5,
        min_size: int = 20,
        max_size: int = 60
    ) -> np.ndarray:
        """Add random rectangular occlusions (missing data patches)"""
        img_array = image.copy()
        h, w = img_array.shape[:2]

        for _ in range(num_patches):
            patch_h = random.randint(min_size, max_size)
            patch_w = random.randint(min_size, max_size)
            y = random.randint(0, max(1, h - patch_h))
            x = random.randint(0, max(1, w - patch_w))
            color = random.choice([0, 255, random.randint(100, 200)])
            img_array[y:y+patch_h, x:x+patch_w] = color

        return img_array

    def add_random_masks(
        self,
        image: np.ndarray,
        num_masks: int = 3
    ) -> np.ndarray:
        """Add random circular masks (simulating sensor defects or obstructions)"""
        img_array = image.copy()
        h, w = img_array.shape[:2]

        for _ in range(num_masks):
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            radius = random.randint(10, min(h, w) // 8)
            color = random.choice([0, 255, random.randint(100, 200)])
            cv2.circle(img_array, (center_x, center_y), radius, int(color), -1)

        return img_array

    def add_scratches(
        self,
        image: np.ndarray,
        num_scratches: int = 5
    ) -> np.ndarray:
        """Add realistic glass/lens scratches with color distortion"""
        img_array = image.copy()
        h, w = image.shape[:2]
        overlay = img_array.copy()

        for _ in range(num_scratches):
            scratch_type = random.choice(['straight', 'curved'])
            
            if scratch_type == 'straight':
                x1, y1 = random.randint(0, w), random.randint(0, h)
                x2, y2 = random.randint(0, w), random.randint(0, h)
                thickness = random.randint(1, 4)
                color = random.choice([
                    (255, 255, 255),
                    (200, 200, 200),
                    (255, 200, 200),
                    (200, 255, 200)
                ])
                cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
                
            else:  # curved
                x1, y1 = random.randint(0, w), random.randint(0, h)
                x2, y2 = random.randint(0, w), random.randint(0, h)
                ctrl_x = (x1 + x2) // 2 + random.randint(-50, 50)
                ctrl_y = (y1 + y2) // 2 + random.randint(-50, 50)
                
                points = []
                for t in np.linspace(0, 1, 20):
                    bx = int((1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2)
                    by = int((1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2)
                    points.append((bx, by))
                
                thickness = random.randint(1, 3)
                color = random.choice([(255, 210, 210), (210, 255, 210), (210, 210, 255)])
                
                for i in range(len(points) - 1):
                    pt1, pt2 = points[i], points[i+1]
                    if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                        0 <= pt2[0] < w and 0 <= pt2[1] < h):
                        cv2.line(overlay, pt1, pt2, color, thickness)

        alpha = random.uniform(0.3, 0.7)
        img_array = cv2.addWeighted(overlay, alpha, img_array, 1 - alpha, 0)
        return img_array

    def add_rain_drops(
        self,
        image: np.ndarray,
        num_drops: int = 30,
        drop_size_range: Tuple[int, int] = (10, 50),
        streak_prob: float = 0.4
    ) -> np.ndarray:
        """Add realistic rain drops with refraction and highlights"""
        img_array = image.astype(np.float32)
        h, w, _ = img_array.shape

        height_map = np.zeros((h, w), dtype=np.float32)
        drop_mask = np.zeros((h, w), dtype=np.uint8)

        for _ in range(num_drops):
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            base_radius = random.randint(drop_size_range[0], drop_size_range[1])
            is_streak = random.random() < streak_prob

            temp_mask = np.zeros((h, w), dtype=np.uint8)

            if is_streak:
                streak_len = random.randint(base_radius * 2, base_radius * 5)
                angle = np.deg2rad(90 + random.uniform(-10, 10))
                head_x = int(center_x + streak_len * np.cos(angle))
                head_y = int(center_y + streak_len * np.sin(angle))

                steps = 20
                for i in range(steps):
                    t = i / float(steps - 1)
                    curr_x = int(center_x * (1-t) + head_x * t)
                    curr_y = int(center_y * (1-t) + head_y * t)
                    curr_r = int(base_radius * (0.2 + 0.8 * t))
                    jitter = int(random.gauss(0, 1)) if t < 0.8 else 0
                    cv2.circle(temp_mask, (curr_x + jitter, curr_y), curr_r, 255, -1)
            else:
                axes = (random.randint(int(base_radius*0.9), base_radius),
                       random.randint(base_radius, int(base_radius*1.1)))
                cv2.ellipse(temp_mask, (center_x, center_y), axes, 
                           random.randint(0, 360), 0, 360, 255, -1)

            current_drop_height = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)
            max_val = current_drop_height.max()
            if max_val > 0:
                current_drop_height /= max_val

            roi = temp_mask > 0
            height_map[roi] = np.maximum(height_map[roi], current_drop_height[roi])
            drop_mask[roi] = 255

        height_map = cv2.GaussianBlur(height_map, (15, 15), 4.0)
        height_map = height_map * (drop_mask > 0)

        grad_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

        refraction_strength = 25.0
        grid_y, grid_x = _coordinate_grids(h, w)
        map_x = grid_x - grad_x * refraction_strength
        map_y = grid_y - grad_y * refraction_strength

        distorted_img = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_REFLECT)

        light_dir = np.array([-1.0, -2.0])
        light_dir /= np.linalg.norm(light_dir)

        specular = np.maximum(grad_x * light_dir[0] + grad_y * light_dir[1], 0)
        specular = np.clip(np.power(specular, 5) * 255, 0, 255)

        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        shadow_map = np.clip(1.0 - edge_magnitude * 0.3, 0.5, 1.0)

        output_img = img_array.copy()
        mask_bool = drop_mask > 0

        output_img[mask_bool] = distorted_img[mask_bool]
        output_img[mask_bool] = output_img[mask_bool] * shadow_map[mask_bool][:, None]
        highlight_val = specular[mask_bool][:, None]
        output_img[mask_bool] = np.clip(output_img[mask_bool] + highlight_val, 0, 255)

        return output_img.astype(np.uint8)

    def apply_degradation(
        self,
        image: np.ndarray,
        degradation_config: Optional[Dict[str, bool]] = None
    ) -> np.ndarray:
        """
        Apply random degradations to an image.
        degradation_config: dict with keys for each degradation type.
        """
        degraded = image.copy()
        
        if degradation_config is None:
            degradation_config = {k: True for k in DEGRADATION_PROBS.keys()}

        # Define degradation handlers
        degradation_handlers = {
            'gaussian_noise': lambda img: self.add_gaussian_noise(img, var=random.uniform(*NOISE_VAR_RANGE)),
            'salt_pepper_noise': lambda img: self.add_salt_pepper_noise(img, amount=random.uniform(*SALT_PEPPER_AMOUNT_RANGE)),
            'blur_gaussian': lambda img: self.apply_blur(img, 'gaussian', random.randint(*BLUR_KERNEL_RANGE)),
            'blur_motion': lambda img: self.apply_blur(img, 'motion', random.randint(*MOTION_BLUR_KERNEL_RANGE)),
            'occlusion': lambda img: self.add_occlusion(img, num_patches=random.randint(*OCCLUSION_PATCHES_RANGE)),
            'random_masks': lambda img: self.add_random_masks(img, num_masks=random.randint(*RANDOM_MASKS_RANGE)),
            'scratches': lambda img: self.add_scratches(img, num_scratches=random.randint(*SCRATCHES_RANGE)),
            'rain_drops': lambda img: self.add_rain_drops(img, num_drops=random.randint(*RAIN_DROPS_RANGE)),
        }

        # Apply degradations based on configuration and probability
        for deg_type, handler in degradation_handlers.items():
            if degradation_config.get(deg_type) and random.random() < DEGRADATION_PROBS[deg_type]:
                degraded = handler(degraded)

        return degraded


class ImageProcessingService:
    """
    High-level service for image augmentation and degradation.
    Provides convenience methods for working with bytes and PIL Images.
    """
    
    def __init__(self):
        self.augmentor = GroundTruthAugmentor()
        self.degradator = ImageDegradator()
        logger.info("✓ ImageProcessingService initialized")
    
    @staticmethod
    def bytes_to_array(image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to numpy array"""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return np.array(image)
    
    @staticmethod
    def array_to_bytes(image_array: np.ndarray, format: str = 'PNG', quality: int = 95) -> bytes:
        """Convert numpy array to image bytes"""
        image = Image.fromarray(image_array.astype(np.uint8))
        buf = io.BytesIO()
        if format.upper() == 'JPEG':
            image.save(buf, format='JPEG', quality=quality, optimize=True)
        else:
            image.save(buf, format='PNG')
        return buf.getvalue()
    
    def augment_image(
        self,
        image_bytes: bytes,
        stretch: bool = True,
        rotation: bool = True,
        zoom: bool = True,
        color: bool = True
    ) -> bytes:
        """Apply augmentation to image bytes and return processed bytes"""
        try:
            img_array = self.bytes_to_array(image_bytes)
            augmented = self.augmentor.apply_augmentation(
                img_array,
                apply_stretch=stretch,
                apply_rotation=rotation,
                apply_zoom=zoom,
                apply_color=color
            )
            return self.array_to_bytes(augmented)
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            raise
    
    def degrade_image(
        self,
        image_bytes: bytes,
        degradation_config: Optional[Dict[str, bool]] = None
    ) -> bytes:
        """Apply degradation to image bytes and return processed bytes"""
        try:
            img_array = self.bytes_to_array(image_bytes)
            degraded = self.degradator.apply_degradation(img_array, degradation_config)
            return self.array_to_bytes(degraded)
        except Exception as e:
            logger.error(f"Degradation failed: {e}")
            raise
    
    def create_preview_variants(
        self,
        image_bytes: bytes,
        num_variants: int = 4
    ) -> List[bytes]:
        """Create multiple augmented/degraded variants for preview"""
        variants = []
        img_array = self.bytes_to_array(image_bytes)
        
        variant_generators = [
            lambda: img_array,  # Original
            lambda: self.augmentor.apply_augmentation(img_array),  # Augmented
            lambda: self.degradator.apply_degradation(img_array),  # Degraded
            lambda: self.degradator.apply_degradation(self.augmentor.apply_augmentation(img_array)),  # Aug+Deg
        ]

        for i, generator in enumerate(variant_generators):
            if i < num_variants:
                variants.append(self.array_to_bytes(generator()))
        
        return variants


# Global instance
image_processing_service = ImageProcessingService()
