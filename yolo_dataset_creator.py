"""
YOLO Dataset Creator Module with Normalization
===============================================
A class-based module for creating YOLO datasets from GeoTIFF + Shapefile
with P2-P98 Min-Max normalization support.

Usage:
------
from yolo_dataset_creator import YoloDatasetFromShp

creator = YoloDatasetFromShp(
    input_tif="path/to/thermal_stack.tif",
    input_shp="path/to/labels.shp",
    output_dir="path/to/output",
    crop_size=640,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    class_names=["ifa_mound"],
    normalize=True,  # Enable normalization
    norm_mode="global",  # or "local" or None
    random_seed=42
)

creator.create_dataset()
"""

import os
import random
import numpy as np
import geopandas as gpd
from osgeo import gdal
from shapely.geometry import box
from typing import List, Tuple, Optional, Literal

gdal.UseExceptions()


class YoloDatasetFromShp:
    """
    Create YOLO-format datasets from thermal stacks and shapefiles.
    
    Attributes:
        input_tif (str): Path to input GeoTIFF file
        input_shp (str): Path to input shapefile with labels
        output_dir (str): Directory for output dataset
        crop_size (int): Size of output crops (default: 640)
        train_ratio (float): Training set ratio (default: 0.8)
        valid_ratio (float): Validation set ratio (default: 0.1)
        test_ratio (float): Test set ratio (default: 0.1)
        class_names (List[str]): List of class names (default: ["object"])
        normalize (bool): Apply P2-P98 normalization (default: False)
        norm_mode (str): "global" or "local" normalization (default: "global")
        random_seed (int): Random seed for reproducibility (default: 42)
    """
    
    def __init__(
        self,
        input_tif: str,
        input_shp: str,
        output_dir: str,
        crop_size: int = 640,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        class_names: Optional[List[str]] = None,
        normalize: bool = False,
        norm_mode: Literal["global", "local"] = "global",
        random_seed: int = 42
    ):
        """
        Initialize YOLO dataset creator.
        
        Args:
            input_tif: Path to multi-band GeoTIFF image
            input_shp: Path to shapefile with polygon labels
            output_dir: Output directory for YOLO dataset
            crop_size: Size of output image crops (pixels)
            train_ratio: Proportion of data for training (0-1)
            valid_ratio: Proportion of data for validation (0-1)
            test_ratio: Proportion of data for testing (0-1)
            class_names: List of class names for detection
            normalize: Whether to apply P2-P98 normalization
            norm_mode: "global" (whole image) or "local" (per-crop)
            random_seed: Random seed for reproducible splits
        """
        self.input_tif = input_tif
        self.input_shp = input_shp
        self.output_dir = output_dir
        self.crop_size = crop_size
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.class_names = class_names or ["object"]
        self.normalize = normalize
        self.norm_mode = norm_mode
        self.random_seed = random_seed
        
        # Validate ratios
        total_ratio = train_ratio + valid_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio:.3f}"
            )
        
        # Validate normalization mode
        if normalize and norm_mode not in ["global", "local"]:
            raise ValueError(
                f"norm_mode must be 'global' or 'local', got '{norm_mode}'"
            )
        
        # Set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Internal state
        self._ds = None
        self._gdf = None
        self._samples = []
        self._global_percentiles = None  # Store global P2/P98 values
        self._stats = {
            'total_polygons': 0,
            'valid_crops': 0,
            'rejected_crops': 0,
            'train_samples': 0,
            'valid_samples': 0,
            'test_samples': 0
        }
    
    def _world_to_pixel(self, x: float, y: float, gt: Tuple) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        px = (x - gt[0]) / gt[1]
        py = (y - gt[3]) / gt[5]
        return int(px), int(py)
    
    def _pixel_to_world(self, px: int, py: int, gt: Tuple) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        x = gt[0] + px * gt[1]
        y = gt[3] + py * gt[5]
        return x, y
    
    def _bbox_to_yolo(
        self, 
        xmin: float, 
        ymin: float, 
        xmax: float, 
        ymax: float, 
        img_w: int, 
        img_h: int
    ) -> Tuple[float, float, float, float]:
        """Convert bounding box to YOLO format (normalized)."""
        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        return x_center, y_center, width, height
    
    def _compute_global_percentiles(self):
        """Compute P2 and P98 percentiles for the entire image."""
        if not self.normalize or self.norm_mode != "global":
            return
        
        print("📊 Computing global P2/P98 percentiles...")
        
        bands = self._ds.RasterCount
        width = self._ds.RasterXSize
        height = self._ds.RasterYSize
        
        self._global_percentiles = []
        
        for b in range(1, bands + 1):
            band = self._ds.GetRasterBand(b)
            
            # Sample image for large files (use every 10th pixel)
            if width * height > 10_000_000:
                step = 10
                sample = band.ReadAsArray(
                    0, 0, width, height
                )[::step, ::step].flatten()
            else:
                sample = band.ReadAsArray().flatten()
            
            # Remove potential nodata values
            sample = sample[np.isfinite(sample)]
            
            p2 = np.percentile(sample, 2)
            p98 = np.percentile(sample, 98)
            
            self._global_percentiles.append((p2, p98))
            print(f"   • Band {b}: P2={p2:.2f}, P98={p98:.2f}")
        
        print()
    
    def _normalize_crop(self, crop_data: np.ndarray) -> np.ndarray:
        """
        Apply P2-P98 Min-Max normalization to crop data.
        
        Args:
            crop_data: Array of shape (bands, height, width)
        
        Returns:
            Normalized array scaled to [0, 255] as uint8
        """
        if not self.normalize:
            return crop_data
        
        normalized = np.zeros_like(crop_data, dtype=np.float32)
        
        for b in range(crop_data.shape[0]):
            band_data = crop_data[b].astype(np.float32)
            
            if self.norm_mode == "global":
                # Use pre-computed global percentiles
                p2, p98 = self._global_percentiles[b]
            else:  # local
                # Compute percentiles for this specific crop
                p2 = np.percentile(band_data, 2)
                p98 = np.percentile(band_data, 98)
            
            # Min-Max normalization
            if p98 > p2:
                normalized[b] = np.clip(
                    (band_data - p2) / (p98 - p2) * 255.0,
                    0, 255
                )
            else:
                # Handle edge case where p2 == p98
                normalized[b] = band_data
        
        return normalized.astype(np.uint8)
    
    def _create_directory_structure(self):
        """Create YOLO dataset directory structure."""
        for split in ["train", "valid", "test"]:
            os.makedirs(
                os.path.join(self.output_dir, split, "images"), 
                exist_ok=True
            )
            os.makedirs(
                os.path.join(self.output_dir, split, "labels"), 
                exist_ok=True
            )
    
    def _create_data_yaml(self):
        """Create YOLO configuration file."""
        yaml_content = f"""# YOLO Dataset Configuration
# Auto-generated from tif + shapefile

train: train/images
val: valid/images
test: test/images

nc: {len(self.class_names)}
names: {self.class_names}

# Normalization settings
normalized: {self.normalize}
norm_mode: {self.norm_mode if self.normalize else 'none'}
"""
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
    
    def _load_data(self):
        """Load GeoTIFF and shapefile data."""
        print("📂 Loading input data...")
        
        # Load GeoTIFF
        self._ds = gdal.Open(self.input_tif)
        if self._ds is None:
            raise FileNotFoundError(f"Cannot open: {self.input_tif}")
        
        width = self._ds.RasterXSize
        height = self._ds.RasterYSize
        bands = self._ds.RasterCount
        gt = self._ds.GetGeoTransform()
        
        print(f"   • Image size: {width} x {height}")
        print(f"   • Bands: {bands}")
        print(f"   • Resolution: {gt[1]:.4f} x {abs(gt[5]):.4f}")
        
        # Load shapefile
        self._gdf = gpd.read_file(self.input_shp)
        print(f"   • Total polygons: {len(self._gdf)}")
        
        # Get image bounds and filter polygons
        minx, maxy = gt[0], gt[3]
        maxx = minx + gt[1] * width
        miny = maxy + gt[5] * height
        image_bounds = box(minx, miny, maxx, maxy)
        
        self._gdf = self._gdf[
            self._gdf.geometry.intersects(image_bounds)
        ].copy()
        
        self._stats['total_polygons'] = len(self._gdf)
        print(f"   • Polygons in bounds: {self._stats['total_polygons']}")
        print(f"   • Normalization: {self.norm_mode if self.normalize else 'disabled'}\n")
        
        # Compute global percentiles if needed
        self._compute_global_percentiles()
    
    def _generate_crops(self):
        """Generate image crops centered on labeled objects."""
        print(f"🔨 Generating {self.crop_size}x{self.crop_size} crops...")
        
        width = self._ds.RasterXSize
        height = self._ds.RasterYSize
        bands = self._ds.RasterCount
        gt = self._ds.GetGeoTransform()
        
        for idx, row in self._gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or geom.geom_type != 'Polygon':
                continue
            
            # Get polygon centroid in pixel coordinates
            centroid = geom.centroid
            cx_px, cy_px = self._world_to_pixel(centroid.x, centroid.y, gt)
            
            # Define crop window centered on polygon
            x1 = max(0, cx_px - self.crop_size // 2)
            y1 = max(0, cy_px - self.crop_size // 2)
            x2 = min(width, x1 + self.crop_size)
            y2 = min(height, y1 + self.crop_size)
            
            # Adjust if near edge
            x1 = max(0, x2 - self.crop_size)
            y1 = max(0, y2 - self.crop_size)
            
            # Verify crop size
            if (x2 - x1) != self.crop_size or (y2 - y1) != self.crop_size:
                self._stats['rejected_crops'] += 1
                continue
            
            # Get crop bounds in world coordinates
            crop_minx, crop_maxy = self._pixel_to_world(x1, y1, gt)
            crop_maxx, crop_miny = self._pixel_to_world(x2, y2, gt)
            crop_bounds = box(crop_minx, crop_miny, crop_maxx, crop_maxy)
            
            # Find all polygons intersecting this crop
            intersecting = self._gdf[
                self._gdf.geometry.intersects(crop_bounds)
            ].copy()
            
            if intersecting.empty:
                continue
            
            # Read image data for all bands
            crop_data = []
            for b in range(1, bands + 1):
                band_arr = self._ds.GetRasterBand(b).ReadAsArray(
                    x1, y1, self.crop_size, self.crop_size
                )
                crop_data.append(band_arr)
            
            crop_data = np.stack(crop_data, axis=0)
            
            # Apply normalization
            crop_data = self._normalize_crop(crop_data)
            
            # Convert polygons to YOLO format
            yolo_labels = []
            for _, poly_row in intersecting.iterrows():
                poly = poly_row.geometry.intersection(crop_bounds)
                if poly.is_empty or poly.geom_type != 'Polygon':
                    continue
                
                # Get bounding box in pixel coordinates relative to crop
                coords = list(poly.exterior.coords)
                pixel_coords = [
                    self._world_to_pixel(x, y, gt) for x, y in coords
                ]
                
                xs = [p[0] - x1 for p in pixel_coords]
                ys = [p[1] - y1 for p in pixel_coords]
                
                xmin = max(0, min(xs))
                xmax = min(self.crop_size, max(xs))
                ymin = max(0, min(ys))
                ymax = min(self.crop_size, max(ys))
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                # Convert to YOLO format
                x_c, y_c, w, h = self._bbox_to_yolo(
                    xmin, ymin, xmax, ymax, 
                    self.crop_size, self.crop_size
                )
                
                # Validate coordinates
                if 0 < x_c < 1 and 0 < y_c < 1 and w > 0 and h > 0:
                    yolo_labels.append(
                        f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                    )
            
            if yolo_labels:
                self._samples.append(
                    (crop_data, yolo_labels, f"sample_{idx:04d}")
                )
                self._stats['valid_crops'] += 1
        
        print(f"   • Valid crops: {self._stats['valid_crops']}")
        print(f"   • Rejected: {self._stats['rejected_crops']}\n")
    
    def _split_dataset(self):
        """Split samples into train/valid/test sets."""
        print("✂️  Splitting dataset...")
        
        random.shuffle(self._samples)
        
        n_total = len(self._samples)
        n_train = int(n_total * self.train_ratio)
        n_valid = int(n_total * self.valid_ratio)
        
        self.train_samples = self._samples[:n_train]
        self.valid_samples = self._samples[n_train:n_train + n_valid]
        self.test_samples = self._samples[n_train + n_valid:]
        
        self._stats['train_samples'] = len(self.train_samples)
        self._stats['valid_samples'] = len(self.valid_samples)
        self._stats['test_samples'] = len(self.test_samples)
        
        print(f"   • Train: {self._stats['train_samples']}")
        print(f"   • Valid: {self._stats['valid_samples']}")
        print(f"   • Test:  {self._stats['test_samples']}\n")
    
    def _save_samples(self):
        """Save samples to disk."""
        print("💾 Saving dataset...")
        
        bands = self._ds.RasterCount
        
        def save_split(split_samples, split_name):
            img_dir = os.path.join(self.output_dir, split_name, "images")
            lbl_dir = os.path.join(self.output_dir, split_name, "labels")
            
            for img_data, labels, sample_id in split_samples:
                # Save image as PNG
                img_path = os.path.join(img_dir, f"{sample_id}.png")
                
                mem_driver = gdal.GetDriverByName("MEM")
                mem_ds = mem_driver.Create(
                    "", self.crop_size, self.crop_size, bands, gdal.GDT_Byte
                )
                
                for b in range(bands):
                    mem_ds.GetRasterBand(b + 1).WriteArray(img_data[b])
                
                png_driver = gdal.GetDriverByName("PNG")
                png_driver.CreateCopy(img_path, mem_ds)
                mem_ds = None
                
                # Save labels
                lbl_path = os.path.join(lbl_dir, f"{sample_id}.txt")
                with open(lbl_path, "w") as f:
                    f.write("\n".join(labels))
        
        save_split(self.train_samples, "train")
        save_split(self.valid_samples, "valid")
        save_split(self.test_samples, "test")
    
    def create_dataset(self):
        """
        Main method to create the YOLO dataset.
        
        Executes the full pipeline:
        1. Load data
        2. Generate crops
        3. Split dataset
        4. Save samples
        5. Create config file
        """
        print(f"\n{'='*60}")
        print("YOLO Dataset Creator - TIF Processing")
        print(f"{'='*60}\n")
        
        try:
            # Execute pipeline
            self._create_directory_structure()
            self._load_data()
            self._generate_crops()
            self._split_dataset()
            self._save_samples()
            self._create_data_yaml()
            
            # Print summary
            print(f"\n{'='*60}")
            print("✅ YOLO Dataset Created Successfully!")
            print(f"{'='*60}")
            print(f"\n📍 Dataset location: {self.output_dir}")
            print(f"📄 Configuration: {os.path.join(self.output_dir, 'data.yaml')}")
            print(f"\n📊 Dataset Statistics:")
            print(f"   • Total polygons: {self._stats['total_polygons']}")
            print(f"   • Valid crops: {self._stats['valid_crops']}")
            print(f"   • Train samples: {self._stats['train_samples']}")
            print(f"   • Valid samples: {self._stats['valid_samples']}")
            print(f"   • Test samples: {self._stats['test_samples']}")
            if self.normalize:
                print(f"   • Normalization: {self.norm_mode} P2-P98")
            print()
            
        finally:
            # Cleanup
            if self._ds is not None:
                self._ds = None
    
    def get_stats(self) -> dict:
        """
        Get dataset creation statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return self._stats.copy()


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    """
    Example usage - Replace paths with your actual files
    """
    
    # Example 1: With global normalization
    creator = YoloDatasetFromShp(
        input_tif="path/to/your/thermal_stack.tif",
        input_shp="path/to/your/labels.shp",
        output_dir="path/to/output/directory",
        crop_size=640,
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        class_names=["ifa_mound"],
        normalize=True,
        norm_mode="global",  # Use global P2/P98 from entire image
        random_seed=42
    )
    
    creator.create_dataset()
    
    # Example 2: With local normalization
    # creator = YoloDatasetFromShp(
    #     input_tif="path/to/your/thermal_stack.tif",
    #     input_shp="path/to/your/labels.shp",
    #     output_dir="path/to/output/directory_local",
    #     crop_size=640,
    #     normalize=True,
    #     norm_mode="local",  # Compute P2/P98 for each crop individually
    #     class_names=["ifa_mound"],
    #     random_seed=42
    # )
    # creator.create_dataset()
    
    # Example 3: Without normalization (original behavior)
    # creator = YoloDatasetFromShp(
    #     input_tif="path/to/your/thermal_stack.tif",
    #     input_shp="path/to/your/labels.shp",
    #     output_dir="path/to/output/directory_raw",
    #     crop_size=640,
    #     normalize=False,  # No normalization
    #     class_names=["ifa_mound"],
    #     random_seed=42
    # )
    # creator.create_dataset()
    
    # Get statistics
    stats = creator.get_stats()
    print(f"Dataset created with {stats['valid_crops']} total samples")
