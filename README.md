# YOLO_dataset_creator

Problem: Sometimes if we have multiple datasets of **large UAV 3 band images (mostly RGB)**  and same labels can be used for all of them. It is challenging to create ***small tiles, .txt labels and .yaml file*** for each large images.

Solution: If we have a **shapefile of labels and dataset directory**, we can use this repo to multiple datasets easily. We can also modify the ***train, test, valid ratio and crop size of images***. 


# Parameters

## YoloDatasetFromShp Class

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_tif` | str | Path to input GeoTIFF image file |
| `input_shp` | str | Path to shapefile containing polygon labels |
| `output_dir` | str | Output directory for YOLO dataset |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crop_size` | int | 640 | Size of output image crops in pixels (e.g., 640×640) |
| `train_ratio` | float | 0.8 | Proportion of data for training (0-1) |
| `valid_ratio` | float | 0.1 | Proportion of data for validation (0-1) |
| `test_ratio` | float | 0.1 | Proportion of data for testing (0-1) |
| `class_names` | list | ["object"] | List of class names for detection |
| `random_seed` | int | 42 | Random seed for reproducible splits |

**Note:** `train_ratio + valid_ratio + test_ratio` must equal 1.0

## Example Usage

```python
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
    random_seed=42
)

creator.create_dataset()
```
# This is the general structure how the code create the output

<img width="623" height="662" alt="image" src="https://github.com/user-attachments/assets/be6bdc0f-7659-46fd-ab44-6d0a7e55847c" />
