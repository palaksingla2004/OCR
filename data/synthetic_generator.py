import os
import numpy as np
import random
import string
import cv2
from PIL import Image, ImageFont, ImageDraw
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class SyntheticTextGenerator:
    """
    Generate synthetic handwritten text data without requiring TextRecognitionDataGenerator.
    """
    def __init__(
        self,
        output_dir: str,
        fonts_dir: Optional[str] = None,
        fonts: Optional[List[str]] = None,
        bg_colors: Optional[List[Tuple[int, int, int]]] = None,
        text_colors: Optional[List[Tuple[int, int, int]]] = None,
        size_range: Tuple[int, int] = (32, 48),
        rotation_range: Tuple[float, float] = (-5, 5),
        blur_range: Tuple[float, float] = (0, 1),
        noise_range: Tuple[float, float] = (0, 0.05),
        random_seed: int = 42
    ):
        """
        Initialize the synthetic text generator.
        
        Args:
            output_dir: Directory to save generated images
            fonts_dir: Directory containing TTF font files (optional)
            fonts: List of font file paths (optional)
            bg_colors: List of background colors as RGB tuples
            text_colors: List of text colors as RGB tuples
            size_range: Range of font sizes (min, max)
            rotation_range: Range of rotation angles in degrees (min, max)
            blur_range: Range of blur factors (min, max)
            noise_range: Range of noise factors (min, max)
            random_seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load fonts
        self.fonts = []
        
        if fonts:
            self.fonts = fonts
        elif fonts_dir:
            for file in os.listdir(fonts_dir):
                if file.endswith(('.ttf', '.otf')):
                    self.fonts.append(os.path.join(fonts_dir, file))
        else:
            # Default fonts that might be available on most systems
            default_fonts = [
                'arial.ttf', 'times.ttf', 'georgia.ttf', 'calibri.ttf',
                'verdana.ttf', 'comic.ttf', 'couri.ttf'
            ]
            
            system_fonts_dirs = [
                "C:\\Windows\\Fonts",  # Windows
                "/usr/share/fonts",    # Linux
                "/System/Library/Fonts"  # macOS
            ]
            
            for fonts_dir in system_fonts_dirs:
                if os.path.exists(fonts_dir):
                    for font in default_fonts:
                        font_path = os.path.join(fonts_dir, font)
                        if os.path.exists(font_path):
                            self.fonts.append(font_path)
        
        if not self.fonts:
            logger.warning("No fonts found! Using standard PIL font.")
            # PIL always has at least one default font
            self.fonts = [None]
            
        # Set colors
        self.bg_colors = bg_colors or [
            (255, 255, 255),  # White
            (245, 245, 245),  # Off-white
            (240, 240, 240),  # Light gray
            (250, 250, 240),  # Ivory
            (255, 250, 250),  # Snow
            (248, 248, 255)   # Ghost white
        ]
        
        self.text_colors = text_colors or [
            (0, 0, 0),        # Black
            (50, 50, 50),     # Dark gray
            (70, 70, 70),     # Medium gray
            (0, 0, 128),      # Navy
            (0, 0, 139),      # Dark blue
            (139, 0, 0)       # Dark red
        ]
        
        self.size_range = size_range
        self.rotation_range = rotation_range
        self.blur_range = blur_range
        self.noise_range = noise_range
    
    def _get_random_text(self, min_length: int = 5, max_length: int = 20) -> str:
        """
        Generate random text.
        
        Args:
            min_length: Minimum length of text
            max_length: Maximum length of text
            
        Returns:
            Randomly generated text
        """
        # Common English words for more realistic text
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
            "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
            "than", "then", "now", "look", "only", "come", "its", "over", "think", "also"
        ]
        
        # Generate a sentence of random length
        length = random.randint(min_length, max_length)
        words = [random.choice(common_words) for _ in range(length)]
        
        # Capitalize first word and add period
        words[0] = words[0].capitalize()
        
        # Sometimes add some punctuation
        for i in range(1, len(words) - 1):
            if random.random() < 0.1:
                words[i] = words[i] + random.choice([",", ";", ":"])
                
        text = " ".join(words) + random.choice([".", "!", "?"])
        return text
    
    def generate_image(self, text: Optional[str] = None, 
                      width: int = 384, height: int = 96) -> Tuple[Image.Image, str]:
        """
        Generate a synthetic handwritten text image.
        
        Args:
            text: Text to render (if None, random text is generated)
            width: Width of the image
            height: Height of the image
            
        Returns:
            Tuple of (PIL Image, text)
        """
        if text is None:
            text = self._get_random_text()
            
        # Create a blank image
        bg_color = random.choice(self.bg_colors)
        image = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Choose a font and size
        font_path = random.choice(self.fonts)
        font_size = random.randint(*self.size_range)
        
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            logger.warning(f"Error loading font {font_path}: {e}")
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        text_color = random.choice(self.text_colors)
        text_width, text_height = draw.textsize(text, font)
        position = ((width - text_width) // 2, (height - text_height) // 2)
        
        # Draw text
        draw.text(position, text, font=font, fill=text_color)
        
        # Convert to numpy array for OpenCV operations
        img_array = np.array(image)
        
        # Apply rotation
        if self.rotation_range[0] != self.rotation_range[1]:
            angle = random.uniform(*self.rotation_range)
            if angle != 0:
                rows, cols, _ = img_array.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                img_array = cv2.warpAffine(img_array, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
        # Apply blur
        if self.blur_range[1] > 0:
            blur = random.uniform(*self.blur_range)
            if blur > 0:
                kernel_size = max(1, int(blur * 3)) * 2 + 1  # Ensure odd kernel size
                img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), blur)
        
        # Apply noise
        if self.noise_range[1] > 0:
            noise_factor = random.uniform(*self.noise_range)
            if noise_factor > 0:
                noise = np.random.normal(0, noise_factor * 255, img_array.shape).astype(np.uint8)
                img_array = cv2.add(img_array, noise)
        
        # Convert back to PIL Image
        synthetic_image = Image.fromarray(img_array)
        
        return synthetic_image, text
    
    def generate_dataset(self, num_samples: int, 
                        width: int = 384, height: int = 96) -> List[Dict[str, str]]:
        """
        Generate a dataset of synthetic handwritten text images.
        
        Args:
            num_samples: Number of samples to generate
            width: Width of the images
            height: Height of the images
            
        Returns:
            List of dictionaries with 'image_path' and 'text' keys
        """
        dataset = []
        
        for i in range(num_samples):
            # Generate image and text
            image, text = self.generate_image(width=width, height=height)
            
            # Save the image
            image_path = os.path.join(self.output_dir, f"synthetic_{i:05d}.png")
            image.save(image_path)
            
            # Add to dataset
            dataset.append({
                'image_path': image_path,
                'text': text
            })
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} synthetic samples")
        
        return dataset
    
    def generate_variations(self, base_text: str, num_variations: int, 
                          width: int = 384, height: int = 96) -> List[Dict[str, str]]:
        """
        Generate variations of the same text.
        
        Args:
            base_text: Base text to create variations of
            num_variations: Number of variations to generate
            width: Width of the images
            height: Height of the images
            
        Returns:
            List of dictionaries with 'image_path' and 'text' keys
        """
        variations = []
        
        for i in range(num_variations):
            # Generate image with the same text
            image, _ = self.generate_image(text=base_text, width=width, height=height)
            
            # Save the image
            image_path = os.path.join(self.output_dir, f"var_{i:05d}_{base_text[:10].replace(' ', '_')}.png")
            image.save(image_path)
            
            # Add to dataset
            variations.append({
                'image_path': image_path,
                'text': base_text
            })
        
        return variations
    
if __name__ == "__main__":
    # Example usage
    generator = SyntheticTextGenerator(output_dir="synthetic_data")
    dataset = generator.generate_dataset(num_samples=100)
    print(f"Generated {len(dataset)} synthetic samples") 