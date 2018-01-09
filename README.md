# AI - ESA - Image Classification and Localization

## Submitted by:
* 01FB15ECS097	Durga Akhil M.
* 01FB15ECS101	G. Sameer
* 01FB15ECS104	Ganesh K.
* 01FB15ECS366	Rahul R. Bharadwaj

## Requirements:
* Python 3.x.x
* Keras 2.1.2
* TensorFlow

## Directory Structure
#### Please read the comments in parentheses () 
.
├──	Report.pdf (The report)
├── Classification-Localisation (Dataset is in data/ and val/ folders)
│   ├── inceptionv3 (Model crashed due to insufficient GPU memory)
│   │   ├── inceptionv3-v1.py
│   │   ├── models (Models are saved here)
│   │   ├── resize.py
│   │   ├── train/
│   │   └── val/
│   ├── old_code_vgg16 (Model crashed due to insufficient GPU memory)
│   │   ├── .
│   │   ├── .
│   │   ├── models (Models are saved here)
│   │   ├── RESULTS_1000_1000 (Output after training for 8.5 hours)
│   │   └── train_val_model.py
│   ├── sliding_window 
│   │   ├── 2007_000243.jpg
│   │   ├── OneSizeWindow.mp4 (Running on trained model for One Size)
│   │   ├── slide.py
│   │   ├── test_cnn_sw.py (Modela and input image as command line args)
│   │   └── VaryingSizeWindow.mp4 (Simulation for all sizes)
│   └── vgg16 (Main working model)
│       ├── models (Models are saved here)
│       ├── resize.py
│       ├── train/
│       ├── val/
│       └── vgg16-v1.py (run this, saves model after each epoch)
├── README.md
└── TransferLearning
    ├── .
    ├── .
    ├── .
    ├── test_frcnn.py (Run as $ python3 test_frcnn.py -p test/)
    └── train_frcnn.py 


#### Demonstration videos are in : ```Classification-Localisation/sliding_window/```

## In TransferLearning/ 
Before running download the following into this folder:

https://drive.google.com/open?id=0B2XPZBSClSzhNmlMb3AzNW5YVW8
https://drive.google.com/open?id=0B2XPZBSClSzhdENkRmM1alVsN3c


## References
https://github.com/yhenon/keras-frcnn