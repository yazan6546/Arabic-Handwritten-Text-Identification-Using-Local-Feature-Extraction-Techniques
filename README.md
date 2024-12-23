# Arabic Handwritten Text Identification Using Local Feature Extraction Techniques

This repository contains the implementation of various techniques for identifying Arabic handwritten text using local feature extraction methods such as ORB and SIFT. The project includes functionalities for evaluating models, processing directories, and visualizing results.

## Table of Contents

- Introduction
- Features
- Installation
- Usage
- Project Structure
- Contributing
- License

## Introduction

Handwritten text recognition is a challenging task, especially for languages like Arabic with complex script. This project aims to identify Arabic handwritten text using local feature extraction techniques, specifically ORB (Oriented FAST and Rotated BRIEF) and SIFT (Scale-Invariant Feature Transform).

## Features

- Evaluate the accuracy of ORB and SIFT models on a test dataset
- Process directories and predict accuracies using SIFT and ORB pipelines
- Visualize original and preprocessed images
- Plot grouped bar charts for accuracy
- Present data in tables for accuracy and keypoints

## Installation

To get started with this project, ensure you have the following dependencies installed:

- Python >= 3.6
- Jupyter Notebook
- OpenCV
- Pandas
- Matplotlib

You can install the required packages using:

```bash
pip install -r requirements.txt
```
## Usage

1. Clone the repository:
```bash
git clone https://github.com/yazan6546/Arabic-Handwritten-Text-Identification-Using-Local-Feature-Extraction-Techniques.git
cd Arabic-Handwritten-Text-Identification-Using-Local-Feature-Extraction-Techniques
```

2. Run Jupyter Notebook to explore the code:
```bash
jupyter notebook
```
3. Open and run the `main.ipynb` notebook to see the implementation and results.

## Project Structure

        
    ├── classes/
    │   ├── clusterer.py               # Defines the Clusterer class for KMeans clustering and creating histograms of clustered features
    │   ├── feature_extractor.py       # Defines the FeatureExtractor class for extracting features using ORB or SIFT methods
    │   └── idf_transformer.py         # Defines the IDFTransformer class for applying Inverse Document Frequency (IDF) transformation to histograms
    ├── data/                          # Directory for storing datasets
    ├── models/                        # Directory for storing trained models
    ├── plots/                         # Directory for storing generated plots
    ├── tables/                        # Contains tables for presenting data
    │   ├── 0.tex
    │   ├── rotate_90.tex
    │   ├── accuracy_table.tex
    │   ├── noise_noise_10.tex
    │   ├── average_keypoints.tex
    │   └── scaling_scale_0_5.tex
    ├── utilities/                     # Contains utility scripts for evaluation, processing, and plotting
    │   ├── plot.py                    # Utility script for plotting images and grouped bar charts
    │   ├── utils.py                   # Utility script for loading images, preprocessing, and calculating keypoints
    │   ├── process.py                 # Utility script for processing directories and calculating accuracies
    │   ├── modify.py                  # Utility script for applying modifications like rotations, noise, and scaling
    │   └── evaluate.py                # Utility script for evaluating the accuracy of ORB and SIFT models
    ├── main.ipynb                     # The main Jupyter Notebook for demonstrating the functionality
    ├── .gitignore
    ├── LICENSE
    └── README.md

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
