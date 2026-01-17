# YOLO_dataset_creator
If you have a shapefile of bounding boxes and a tif image and want to create YOLO dataset. We can use this.


C:\User\
├── composites\
│   └── Input_directory\
│       ├── image1.tif
│       ├── image2.tif
│       └── image3.tif
├── labels.shp
└── YOLO_Datasets\          ← All outputs here
    ├── Dataset_image1\
    │   ├── train\
    │   ├── valid\
    │   ├── test\
    │   └── data.yaml
    ├── Dataset_image2\
    │   ├── train\
    │   ├── valid\
    │   ├── test\
    │   └── data.yaml
    └── Dataset_image3\
        ├── train\
        ├── valid\
        ├── test\
        └── data.yaml
