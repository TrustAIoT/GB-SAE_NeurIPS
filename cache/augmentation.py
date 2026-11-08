"""Extensible PIL-image augmentation; original slight-augmentation math retained."""
import torch
from PIL import Image
from torchvision import transforms

from .utils import deterministic_seed, set_global_seed, image_random_state, import_symbol


class DeterministicSlightAugmentation:
    """Sample crop/flip/color/blur transforms using the caller-controlled torch RNG.
    
    The transform itself does not seed the RNG. Cache callers derive an image/
    view seed and call set_global_seed before applying it. Color jitter is chosen
    with probability 0.4; weak desaturation has probability 0.6*0.2=0.12 because
    its separate draw occurs only after the first branch fails.
    """
    def __init__(self, image_size=224):
        self.image_size = image_size

        # --- stronger but still object-centric crop ---
        self.random_resized_crop = transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),      
            ratio=(0.85, 1.15)
        )

        # --- color jitter (normal) ---
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        )

        # --- weak desaturation (instead of grayscale) ---
        self.weak_desat = transforms.ColorJitter(
            saturation=(0.4, 0.7)  
        )

        # --- flips ---
        self.hflip = transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = transforms.RandomVerticalFlip(p=1.0)

        # --- blur ---
        self.gaussian_blur = transforms.GaussianBlur(
            kernel_size=23,
            sigma=(0.15, 1.3)
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. always-on crop
        img = self.random_resized_crop(img)

        # 2. horizontal flip (50%)
        if torch.rand(1).item() < 0.5:
            img = self.hflip(img)

        # 3. vertical flip (15%)
        if torch.rand(1).item() < 0.15:
            img = self.vflip(img)

        # 4. color jitter OR weak desaturation (lower prob)
        if torch.rand(1).item() < 0.4:
            img = self.color_jitter(img)
        elif torch.rand(1).item() < 0.2:
            img = self.weak_desat(img)

        # 5. gaussian blur (30%)
        if torch.rand(1).item() < 0.3:
            img = self.gaussian_blur(img)

        return img


AUGMENTATION_REGISTRY = {"slight": DeterministicSlightAugmentation}


def register_augmentation(name: str, factory):
    """Register factory(**kwargs)->callable(PIL.Image)->PIL.Image."""
    if name in AUGMENTATION_REGISTRY:
        raise ValueError(f"Augmentation already registered: {name}")
    AUGMENTATION_REGISTRY[name] = factory


def build_augmentation(options: dict):
    if not options.get("enabled", False):
        return None
    name = options.get("type", "slight")
    factory = AUGMENTATION_REGISTRY.get(name)
    if factory is None:
        factory = import_symbol(name)
    return factory(**options.get("kwargs", {}))


def generate_views(image, image_name: str, count: int, base_seed: int, augmenter):
    """Return view-id->PIL image; N=1 augmentation uses only view 1."""
    if augmenter is None:
        return {0: image}
    views = {0: image} if count > 1 else {}
    view_ids = range(1, count) if count > 1 else [1]
    for view_id in view_ids:
        with image_random_state(deterministic_seed(image_name, view_id, base_seed)):
            value = augmenter(image.copy())
        if not isinstance(value, Image.Image):
            raise TypeError("Augmentation must return PIL.Image; wrap external tensor/dict transforms in an adapter")
        views[view_id] = value
    return views
