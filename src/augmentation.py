import glob
import math
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from multiprocessing import cpu_count
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def show_images(augmentations: dict[str, Image.Image]) -> None:
    num_cols: int = 3
    num_rows: int = math.ceil(len(augmentations) / num_cols)
    fig: Any
    axes: Any
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
    axes = axes.flatten()

    idx: int
    name: str
    img: Image.Image
    for idx, (name, img) in enumerate(augmentations.items()):
        axes[idx].imshow(img)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(len(augmentations), num_cols * num_rows):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


@lru_cache(maxsize=32)
def _motion_blur_kernel(kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel: np.ndarray = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[int((kernel_size - 1) / 2), :] = 1.0
    kernel /= kernel_size
    return kernel


@lru_cache(maxsize=8)
def _coordinate_grids(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    return grid_y, grid_x


DEFAULT_JPEG_QUALITY: int = 95


def _save_jpeg(array: np.ndarray, output_path: str, quality: int = DEFAULT_JPEG_QUALITY) -> None:
    image = Image.fromarray(array.astype(np.uint8))
    image.save(output_path, format='JPEG', quality=quality, optimize=True)


class GroundTruthAugmentor:
    """
    Ground truth augmentation class for geometric and colorimetric transformations.
    Creates augmented clean images without degradation or black gaps.
    """

    def demo(self, image: Image.Image) -> None:
        # Create different augmented versions
        augmentations: dict[str, Image.Image] = {
            'Original': image,
            'Stretch': Image.fromarray(self.apply_stretch(np.array(image))),
            'Rotation': Image.fromarray(self.apply_rotation(np.array(image), angle=15)),
            'Zoom': Image.fromarray(self.apply_zoom(np.array(image), zoom_factor=1.2)),
            'Color Alteration': Image.fromarray(self.alter_colorimetry(np.array(image))),
            'Combined': self.apply_augmentation(np.array(image))
        }
        show_images(augmentations)

    def apply_stretch(self, image: np.ndarray, stretch_factor_x: float | None = None, stretch_factor_y: float | None = None) -> np.ndarray:
        """
        Apply stretch transformation without black gaps.
        Uses interpolation and crops back to original size.
        """
        if stretch_factor_x is None:
            stretch_factor_x = random.uniform(0.8, 1.2)
        if stretch_factor_y is None:
            stretch_factor_y = random.uniform(0.8, 1.2)

        h: int
        w: int
        h, w = image.shape[:2]

        # Calculate new dimensions
        new_w: int = int(w * stretch_factor_x)
        new_h: int = int(h * stretch_factor_y)

        # Stretch the image
        # Ensure image is compatible with OpenCV
        # image = np.array(image, dtype=np.uint8)
        # if not image.flags['C_CONTIGUOUS']:
        #     image = np.ascontiguousarray(image)

        stretched: np.ndarray = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Crop or pad to maintain original size
        result: np.ndarray
        if new_w >= w and new_h >= h:
            # Crop from center
            start_x: int = (new_w - w) // 2
            start_y: int = (new_h - h) // 2
            result = stretched[start_y:start_y + h, start_x:start_x + w]
        elif new_w < w and new_h < h:
            # Tile/reflect to fill
            result = cv2.resize(stretched, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Mixed case: crop one dimension, resize other
            result = cv2.resize(stretched, (w, h), interpolation=cv2.INTER_LINEAR)

        return result

    def apply_rotation(self, image: np.ndarray, angle: float | None = None) -> np.ndarray:
        """
        Apply rotation without black gaps using border reflection.
        """
        if angle is None:
            angle = random.uniform(-15, 15)

        h: int
        w: int
        h, w = image.shape[:2]
        center: tuple[int, int] = (w // 2, h // 2)

        # Get rotation matrix
        rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounding dimensions
        abs_cos: float = abs(rotation_matrix[0, 0])
        abs_sin: float = abs(rotation_matrix[0, 1])
        bound_w: int = int(h * abs_sin + w * abs_cos)
        bound_h: int = int(h * abs_cos + w * abs_sin)

        # Adjust rotation matrix for new bounds
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]

        # Rotate with border reflection to avoid black gaps
        rotated: np.ndarray = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h),
                                 borderMode=cv2.BORDER_REFLECT)

        # Crop back to original size from center
        crop_x: int = (bound_w - w) // 2
        crop_y: int = (bound_h - h) // 2
        cropped: np.ndarray = rotated[crop_y:crop_y + h, crop_x:crop_x + w]

        return cropped

    def apply_zoom(self, image: np.ndarray, zoom_factor: float | None = None) -> np.ndarray:
        """
        Apply zoom (crop-based) without black gaps.
        Zoom in: crop and resize. Zoom out: not applicable (would create gaps).
        """
        if zoom_factor is None:
            zoom_factor = random.uniform(1.0, 1.4)  # Only zoom in

        h: int
        w: int
        h, w = image.shape[:2]

        # Zoom in by cropping from center
        new_h: int = int(h / zoom_factor)
        new_w: int = int(w / zoom_factor)

        top: int = (h - new_h) // 2
        left: int = (w - new_w) // 2

        # Crop
        cropped: np.ndarray = image[top:top + new_h, left:left + new_w]

        # Resize back to original size
        zoomed: np.ndarray = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return zoomed

    def alter_colorimetry(self, image: np.ndarray, brightness_factor: float | None = None, contrast_factor: float | None = None,
                         saturation_factor: float | None = None, hue_shift: int | None = None) -> np.ndarray:
        """
        Modify brightness, contrast, saturation, and hue for ground truth augmentation.
        """
        img_array: np.ndarray = image.astype(np.float32)

        if brightness_factor is None:
            brightness_factor = random.uniform(0.7, 1.3)
        img_array = img_array * brightness_factor

        if contrast_factor is None:
            contrast_factor = random.uniform(0.7, 1.3)
        mean = np.mean(img_array, axis=(0, 1), keepdims=True)
        img_array = (img_array - mean) * contrast_factor + mean

        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        # Convert to HSV for saturation and hue
        hsv: np.ndarray = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

        if saturation_factor is None:
            saturation_factor = random.uniform(0.6, 1.4)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor

        if hue_shift is None:
            hue_shift = random.randint(-20, 20)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img_array

    def apply_augmentation(self, image: Image.Image | np.ndarray, return_image: bool = True) -> Image.Image | np.ndarray:
        """
        Apply random ground truth augmentations (geometric + colorimetric).
        This creates new ground truth images without degradation.
        """
        if isinstance(image, Image.Image):
            augmented: np.ndarray = np.array(image)
        else:
            augmented = image.copy()

        augmentation_applied: bool = False

        if random.random() < 0.4:
            augmented = self.apply_stretch(augmented)
            augmentation_applied = True
        if random.random() < 0.4:
            augmented = self.apply_rotation(augmented)
            augmentation_applied = True
        if random.random() < 0.4:
            augmented = self.apply_zoom(augmented)
            augmentation_applied = True
        if random.random() < 0.5:
            augmented = self.alter_colorimetry(augmented)
            augmentation_applied = True

        if not augmentation_applied:
            augmented = self.apply_rotation(augmented)

        return Image.fromarray(augmented) if return_image else augmented


class ImageDegradator:
    """
    Image degradation class for applying realistic degradations to images.
    Used for creating degraded versions of clean images for training restoration models.
    """

    def demo(self, sample_img: Image.Image) -> None:
        img_array = np.array(sample_img)
        augmentations: dict[str, Image.Image] = {
            "Original": sample_img,
            "Gaussian Noise": Image.fromarray(self.add_gaussian_noise(img_array)),
            "Salt and Pepper Noise": Image.fromarray(self.add_salt_pepper_noise(img_array)),
            "Blurred": Image.fromarray(self.apply_blur(img_array)),
            "Occlusion": Image.fromarray(self.add_occlusion(img_array)),
            "Random Masks": Image.fromarray(self.add_random_masks(img_array)),
            "Scratches": Image.fromarray(self.add_scratches(img_array)),
            "Rain Drops": Image.fromarray(self.add_rain_drops(img_array)),
            "Random Degradations": self.apply_degradation(img_array)
        }
        show_images(augmentations)

    def add_gaussian_noise(self, image: np.ndarray, mean: float = 0, var: float = 0.01) -> np.ndarray:
        """Add Gaussian (normal distribution) noise to image"""
        img_float: np.ndarray = image.astype(np.float32) / 255.0
        noise: np.ndarray = np.random.normal(mean, var**0.5, img_float.shape)
        noisy: np.ndarray = np.clip(img_float + noise, 0, 1)
        return (noisy * 255).astype(np.uint8)

    def add_salt_pepper_noise(self, image: np.ndarray, amount: float = 0.05) -> np.ndarray:
        """Add salt and pepper noise (random white and black pixels)"""
        noisy: np.ndarray = image.copy()
        h: int
        w: int
        h, w = image.shape[:2]

        # Salt (white pixels)
        num_salt: int = np.ceil(amount * image.size * 0.5).astype(int)
        coords_salt: list[np.ndarray] = [np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)]
        if image.ndim == 3:
            noisy[coords_salt[0], coords_salt[1], :] = 255
        else:
            noisy[coords_salt[0], coords_salt[1]] = 255

        # Pepper (black pixels)
        num_pepper: int = np.ceil(amount * image.size * 0.5).astype(int)
        coords_pepper: list[np.ndarray] = [np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper)]
        if image.ndim == 3:
            noisy[coords_pepper[0], coords_pepper[1], :] = 0
        else:
            noisy[coords_pepper[0], coords_pepper[1]] = 0

        return noisy

    def apply_blur(self, image: np.ndarray, blur_type: str = 'gaussian', kernel_size: int = 5) -> np.ndarray:
        """Apply various types of blur"""
        blurred: np.ndarray

        if blur_type == 'gaussian':
            if kernel_size % 2 == 0: kernel_size += 1
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == 'motion':
            kernel = _motion_blur_kernel(kernel_size)
            blurred = cv2.filter2D(image, -1, kernel)
        elif blur_type == 'box':
            blurred = cv2.blur(image, (kernel_size, kernel_size))
        else:
            raise ValueError(f"Unknown blur type: {blur_type}")

        return blurred

    def add_occlusion(self, image: np.ndarray, num_patches: int = 5, min_size: int = 20, max_size: int = 60) -> np.ndarray:
        """Add random rectangular occlusions (missing data patches)"""
        img_array: np.ndarray = image.copy()
        h: int
        w: int
        h, w = img_array.shape[:2]

        for _ in range(num_patches):
            patch_h: int = random.randint(min_size, max_size)
            patch_w: int = random.randint(min_size, max_size)
            y: int = random.randint(0, h - patch_h)
            x: int = random.randint(0, w - patch_w)
            color: int = random.choice([0, 255, random.randint(100, 200)])
            img_array[y:y+patch_h, x:x+patch_w] = color

        return img_array

    def add_random_masks(self, image: np.ndarray, num_masks: int = 3) -> np.ndarray:
        """Add random circular masks (simulating sensor defects or obstructions)"""
        img_array: np.ndarray = image.copy()
        h: int
        w: int
        h, w = img_array.shape[:2]

        for _ in range(num_masks):
            center_x: int = random.randint(0, w)
            center_y: int = random.randint(0, h)
            radius: int = random.randint(10, min(h, w) // 8)
            color: int = random.choice([0, 255, random.randint(100, 200)])
            cv2.circle(img_array, (center_x, center_y), radius, int(color), -1)

        return img_array

    def add_scratches(self, image: np.ndarray, num_scratches: int = 5) -> np.ndarray:
        """Add realistic glass/lens scratches with color distortion and curved patterns"""
        img_array: np.ndarray = image.copy()
        h: int
        w: int
        h, w = img_array.shape[:2]
        overlay: np.ndarray = img_array.copy()

        for _ in range(num_scratches):
            scratch_type: str = random.choice(['straight', 'curved', 'arc'])
            if scratch_type == 'straight':
                x1: int = random.randint(0, w)
                y1: int = random.randint(0, h)
                x2: int = random.randint(0, w)
                y2: int = random.randint(0, h)
                thickness: int = random.randint(1, 4)
                scratch_style: str = random.choice(['white', 'gray', 'chromatic'])
                color: tuple[int, int, int]
                if scratch_style == 'white': color = (255, 255, 255)
                elif scratch_style == 'gray':
                    gray_val: int = random.randint(180, 245)
                    color = (gray_val, gray_val, gray_val)
                else: color = random.choice([(255, 200, 200), (200, 255, 200), (200, 200, 255), (255, 255, 200)])
                cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
                if random.random() < 0.3:
                    offset_color: tuple[int, int, int] = random.choice([(255, 150, 150), (150, 255, 150), (150, 150, 255)])
                    offset: int = random.randint(1, 2)
                    cv2.line(overlay, (x1+offset, y1+offset), (x2+offset, y2+offset), offset_color, max(1, thickness-1))
            elif scratch_type == 'curved':
                x1: int = random.randint(0, w)
                y1: int = random.randint(0, h)
                x2: int = random.randint(0, w)
                y2: int = random.randint(0, h)
                ctrl_x: int = (x1 + x2) // 2 + random.randint(-50, 50)
                ctrl_y: int = (y1 + y2) // 2 + random.randint(-50, 50)
                num_points: int = 20; points: list[tuple[int, int]] = []
                for t in np.linspace(0, 1, num_points):
                    bx: int = int((1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2)
                    by: int = int((1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2)
                    points.append((bx, by))
                thickness: int = random.randint(1, 3)
                if random.random() < 0.5:
                    scratch_val: int = random.randint(200, 255)
                    color = (scratch_val, scratch_val, scratch_val)
                else: color = random.choice([(255, 210, 210), (210, 255, 210), (210, 210, 255)])
                for i in range(len(points) - 1):
                    pt1: tuple[int, int] = points[i]
                    pt2: tuple[int, int] = points[i+1]
                    if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h):
                        cv2.line(overlay, pt1, pt2, color, thickness)
                if random.random() < 0.3:
                    offset_color: tuple[int, int, int] = random.choice([(255, 180, 180), (180, 255, 180), (180, 180, 255)])
                    for i in range(len(points) - 1):
                        pt1 = (points[i][0] + 1, points[i][1] + 1)
                        pt2 = (points[i+1][0] + 1, points[i+1][1] + 1)
                        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h):
                            cv2.line(overlay, pt1, pt2, offset_color, 1)
            else:
                center_x: int = random.randint(0, w); center_y: int = random.randint(0, h)
                radius: int = random.randint(30, min(w, h) // 3)
                start_angle: int = random.randint(0, 360); end_angle: int = start_angle + random.randint(30, 180)
                thickness: int = random.randint(1, 3)
                if random.random() < 0.5:
                    arc_val: int = random.randint(200, 255)
                    color = (arc_val, arc_val, arc_val)
                else: color = random.choice([(255, 210, 210), (210, 255, 210), (210, 210, 255)])
                cv2.ellipse(overlay, (center_x, center_y), (radius, radius), 0, start_angle, end_angle, color, thickness)
                if random.random() < 0.4:
                    fringe_color: tuple[int, int, int] = random.choice([(255, 200, 200), (200, 255, 200), (200, 200, 255)])
                    cv2.ellipse(overlay, (center_x + 1, center_y + 1), (radius + 1, radius + 1), 0, start_angle, end_angle, fringe_color, 1)

        alpha: float = random.uniform(0.3, 0.7)
        img_array = cv2.addWeighted(overlay, alpha, img_array, 1 - alpha, 0)
        return img_array

    def add_rain_drops(self, image: np.ndarray, num_drops: int = 30, drop_size_range: tuple[int, int] = (10, 50), streak_prob: float = 0.4) -> np.ndarray:
        """
        Add realistic rain drops with refraction, highlights, and streaks.
        """
        img_array: np.ndarray = image.astype(np.float32)
        h: int
        w: int
        h, w, _ = img_array.shape

        # Create height map for lens-like refraction
        height_map: np.ndarray = np.zeros((h, w), dtype=np.float32)
        drop_mask: np.ndarray = np.zeros((h, w), dtype=np.uint8)

        for _ in range(num_drops):
            center_x: int = random.randint(0, w)
            center_y: int = random.randint(0, h)
            base_radius: int = random.randint(drop_size_range[0], drop_size_range[1])
            is_streak: bool = random.random() < streak_prob

            temp_mask: np.ndarray = np.zeros((h, w), dtype=np.uint8)

            if is_streak:
                # Organic streak generation
                streak_len: int = random.randint(base_radius * 2, base_radius * 5)
                angle: float = np.deg2rad(90 + random.uniform(-10, 10))

                head_x: int = int(center_x + streak_len * np.cos(angle))
                head_y: int = int(center_y + streak_len * np.sin(angle))

                steps: int = 20
                for i in range(steps):
                    t: float = i / float(steps - 1)
                    curr_x: int = int(center_x * (1-t) + head_x * t)
                    curr_y: int = int(center_y * (1-t) + head_y * t)
                    curr_r: int = int(base_radius * (0.2 + 0.8 * t))
                    jitter: int = int(random.gauss(0, 1)) if t < 0.8 else 0
                    cv2.circle(temp_mask, (curr_x + jitter, curr_y), curr_r, 255, -1)
            else:
                # Elliptical drop
                axes: tuple[int, int] = (random.randint(int(base_radius*0.9), base_radius),
                        random.randint(base_radius, int(base_radius*1.1)))
                cv2.ellipse(temp_mask, (center_x, center_y), axes, random.randint(0, 360), 0, 360, 255, -1)

            current_drop_height: np.ndarray = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)
            max_val: float = current_drop_height.max()
            if max_val > 0:
                current_drop_height /= max_val

            roi: np.ndarray = temp_mask > 0
            height_map[roi] = np.maximum(height_map[roi], current_drop_height[roi])
            drop_mask[roi] = 255

        # Smooth the height map
        height_map = cv2.GaussianBlur(height_map, (15, 15), 4.0)
        height_map = height_map * (drop_mask > 0)

        # Calculate refraction
        grad_x: np.ndarray = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
        grad_y: np.ndarray = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

        refraction_strength: float = 25.0
        grid_y, grid_x = _coordinate_grids(h, w)
        map_x: np.ndarray = grid_x - grad_x * refraction_strength
        map_y: np.ndarray = grid_y - grad_y * refraction_strength

        # Apply distortion
        distorted_img: np.ndarray = cv2.remap(img_array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Add lighting effects
        light_dir: np.ndarray = np.array([-1.0, -2.0])
        light_dir /= np.linalg.norm(light_dir)

        specular: np.ndarray = (grad_x * light_dir[0] + grad_y * light_dir[1])
        specular = np.maximum(specular, 0)
        specular = np.power(specular, 5) * 255
        specular = np.clip(specular, 0, 255)

        edge_magnitude: np.ndarray = np.sqrt(grad_x**2 + grad_y**2)
        shadow_map: np.ndarray = np.clip(1.0 - edge_magnitude * 0.3, 0.5, 1.0)

        # Composite
        output_img: np.ndarray = img_array.copy()
        mask_bool: np.ndarray = drop_mask > 0

        output_img[mask_bool] = distorted_img[mask_bool]
        output_img[mask_bool] = output_img[mask_bool] * shadow_map[mask_bool][:, None]
        highlight_val: np.ndarray = specular[mask_bool][:, None]
        output_img[mask_bool] = np.clip(output_img[mask_bool] + highlight_val, 0, 255)

        return output_img.astype(np.uint8)

    def apply_degradation(self, image: Image.Image | np.ndarray, degradation_list: list[tuple[str, float]] | None = None,
                          return_image: bool = True) -> Image.Image | np.ndarray:
        """
        Apply random degradations to an image.
        Degradations: noise, blur, occlusion, glass scratches, rain drops.
        """
        if isinstance(image, Image.Image):
            degraded: np.ndarray = np.array(image)
        else:
            degraded = image.copy()

        if degradation_list is None:
            degradation_list = [
                ('gaussian_noise', 0.3),
                ('salt_pepper_noise', 0.2),
                ('blur_gaussian', 0.4),
                ('blur_motion', 0.3),
                ('occlusion', 0.3),
                ('random_masks', 0.3),
                ('scratches', 0.25),
                ('rain_drops', 0.4),
            ]

        active_degradations = [name for name, probability in degradation_list if random.random() < probability]

        for deg_name in active_degradations:
            if deg_name == 'gaussian_noise':
                degraded = self.add_gaussian_noise(degraded, var=random.uniform(0.005, 0.02))
            elif deg_name == 'salt_pepper_noise':
                degraded = self.add_salt_pepper_noise(degraded, amount=random.uniform(0.01, 0.05))
            elif deg_name == 'blur_gaussian':
                degraded = self.apply_blur(degraded, 'gaussian', random.randint(3, 7))
            elif deg_name == 'blur_motion':
                degraded = self.apply_blur(degraded, 'motion', random.randint(5, 15))
            elif deg_name == 'occlusion':
                degraded = self.add_occlusion(degraded, num_patches=random.randint(2, 5))
            elif deg_name == 'random_masks':
                degraded = self.add_random_masks(degraded, num_masks=random.randint(2, 5))
            elif deg_name == 'scratches':
                degraded = self.add_scratches(degraded, num_scratches=random.randint(3, 8))
            elif deg_name == 'rain_drops':
                degraded = self.add_rain_drops(degraded, num_drops=random.randint(10, 30))

        return Image.fromarray(degraded) if return_image else degraded


def _generate_pairs_for_image(task: tuple[str, str, str, int, int, int]) -> int:
    """Generate clean/degraded pairs for a single image path."""
    img_path, clean_dir, degraded_dir, num_augmentations, num_degradations, jpeg_quality = task
    augmentor = GroundTruthAugmentor()
    degradator = ImageDegradator()

    with Image.open(img_path) as base_img:
        base_img = base_img.convert('RGB')
        base_array = np.array(base_img)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    version_idx: int = 0
    generated_pairs: int = 0
    for _ in range(num_degradations):
        clean_filename = f"{base_name}_v{version_idx}.jpg"
        clean_path = os.path.join(clean_dir, clean_filename)
        base_img.save(clean_path, format='JPEG', quality=jpeg_quality, optimize=True)

        degraded_array = degradator.apply_degradation(base_array, return_image=False)
        degraded_path = os.path.join(degraded_dir, clean_filename)
        _save_jpeg(degraded_array, degraded_path, jpeg_quality)

        generated_pairs += 1
        version_idx += 1


    for _ in range(num_augmentations):
        augmented_array = augmentor.apply_augmentation(base_array, return_image=False)
        for _ in range(num_degradations):
            clean_filename = f"{base_name}_v{version_idx}.jpg"
            clean_path = os.path.join(clean_dir, clean_filename)

            _save_jpeg(augmented_array, clean_path, jpeg_quality)

            degraded_array = degradator.apply_degradation(augmented_array, return_image=False)
            degraded_path = os.path.join(degraded_dir, clean_filename)
            _save_jpeg(degraded_array, degraded_path, jpeg_quality)

            generated_pairs += 1
            version_idx += 1

    return generated_pairs


def generate_dataset(
    image_paths: list[str],
    output_dir: str,
    num_augmentations_per_image: int = 3,
    num_degradations_per_image: int = 1,
    force_regenerate=False,
    n_jobs=None
) -> tuple[str, str, int]:
    """
    2.2.1 Generate Ground Truth Augmentation
    Creates augmented clean images using GroundTruthAugmentor.

    Returns: Tuple[
        str: Clean Directory path,
        str: Degraded Directory path,
        int: Total generated pair
    ]
    """

    print(f"Augmentation Configuration:")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Available CPU cores: {cpu_count()}")
    print(f"  - Ground truth augmentations per image: {num_augmentations_per_image}")
    print(f"  - Degradations per ground truth image: {num_degradations_per_image}")

    clean_dir = os.path.join(output_dir, 'clean')
    degraded_dir = os.path.join(output_dir, 'degraded')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(degraded_dir, exist_ok=True)

    assert num_degradations_per_image > 0, "Can't have zero degradations per image."

    # Check existing
    total_pairs = len(image_paths) * num_degradations_per_image * (num_augmentations_per_image + 1) # 1 for the original images
    clean_existing = glob.glob(os.path.join(clean_dir, '*.jpg'))
    degraded_existing = glob.glob(os.path.join(degraded_dir, '*.jpg'))
    expected = total_pairs

    print(f"Expected size: {expected}")
    print(f"Clean existing: {len(clean_existing)}")
    print(f"Degraded existing: {len(degraded_existing)}")

    if not force_regenerate and len(clean_existing) > 0 and len(degraded_existing) > 0 \
    and len(clean_existing) == expected and len(degraded_existing) == expected:
        print(f"✓ Dataset already exists ({len(clean_existing)} images).")
        return clean_dir, degraded_dir, len(clean_existing)

    if len(clean_existing) > 0 and len(degraded_existing) > 0 \
    and len(clean_existing) != expected and len(degraded_existing) != expected:
        print("Generation was incomplete recreating from zero to avoid mismatched pairs...")
        shutil.rmtree(clean_dir)
        os.makedirs(clean_dir, exist_ok=True)
        shutil.rmtree(degraded_dir)
        os.makedirs(degraded_dir, exist_ok=True)

    if n_jobs is None:
        n_jobs = min(cpu_count(), len(image_paths)) or 1

    print(f"Generating {total_pairs} image pair dataset from {len(image_paths)} images...")
    print(f"Generating {num_augmentations_per_image * num_degradations_per_image} versions per image")

    tasks = [
        (
            img_path,
            clean_dir,
            degraded_dir,
            num_augmentations_per_image,
            num_degradations_per_image,
            DEFAULT_JPEG_QUALITY,
        )
        for img_path in image_paths
    ]

    total_generated: int = 0

    with tqdm(total=len(tasks), desc="Generating pairs") as pbar:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_image = {executor.submit(_generate_pairs_for_image, task): task[0] for task in tasks}

            for future in as_completed(future_to_image):
                img_path = future_to_image[future]
                try:
                    total_generated += future.result()
                except Exception as exc:
                    print(f"Error processing {img_path}: {exc}")
                finally:
                    pbar.update(1)

    return clean_dir, degraded_dir, total_generated


