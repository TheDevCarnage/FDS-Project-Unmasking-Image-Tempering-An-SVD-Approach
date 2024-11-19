# FDS-Project-Unmasking-Image-Tempering-An-SVD-Approach

## Project Overview

Image forgery detection has become an important area of research as the increasing sophistication of digital manipulation tools allows for seamless alteration of images. Such manipulations, like copy-paste tampering or splicing, are often undetectable to the human eye, posing a significant challenge to authenticity verification in legal, journalistic, and security contexts.

This project leverages Singular Value Decomposition (SVD), a mathematical technique that decomposes an image matrix into singular values, to detect image forgery. SVD is particularly useful because image tampering often alters the singular value patterns of an image, leaving detectable traces. By analyzing these changes, this project identifies inconsistent singular values that can indicate potential forgeries.

The core idea is to exploit the inherent differences between the singular values of authentic images and those of tampered images. The method works by analyzing image decomposition and extracting features from these singular values that are most sensitive to tampering. These features are then used to distinguish between real and manipulated images. Additionally, visual techniques like heatmaps are used to highlight tampered areas, offering an interpretable method to pinpoint manipulation.

This project aims to not only detect forged regions but also to provide insights into the underlying causes of inconsistencies introduced by image tampering. By combining SVD-based feature extraction with modern deep learning techniques, this project lays the groundwork for a robust, automated system capable of detecting a wide range of image manipulations.

## Problem Statement

The increasing prevalence of digital image manipulation calls for robust methods to detect tampered images. Traditional image processing methods often fail to capture subtle alterations in an image, especially when the tampering is done seamlessly. This project tackles this problem by applying SVD for feature extraction, followed by classification to identify tampered images. The goal is to build a system that can accurately detect tampered regions in images, and distinguish authentic content from fake or altered images, using an advanced, interpretable approach based on mathematical decomposition.

## Key Features of the Project

- **SVD Decomposition**: Singular Value Decomposition breaks down an image into its singular values, which represent important structural information of the image. Image manipulation often leads to changes in these values, making them an effective tool for tampering detection.

- **Feature Extraction**: By examining the mean, variance, energy ratio, and other statistical metrics of the singular values, we can extract significant features that help identify discrepancies between original and manipulated images. These extracted features provide key insights into potential tampering.

- **Block-based Analysis**: To enhance detection accuracy, the image is divided into smaller blocks (e.g., 16x16 or 32x32). SVD is then applied to each block, and statistical metrics are computed for the top singular values of each block. This localized approach allows for the identification of tampered regions with higher precision.

- **Anomaly Detection**: The project identifies anomalies in the extracted features across different blocks. Significant deviations in singular values, such as sudden drops in variance or high energy ratios, are indicative of tampering. These anomalies are flagged for further investigation.

- **Deep Learning Classification**: To automate the detection process, a Convolutional Neural Network (CNN) is trained to classify images as either authentic or tampered. The CNN uses the extracted features as input, learning the complex relationships between them to make accurate predictions.

- **Visualization**: The tampered regions of the image are highlighted using heatmaps, providing a visual representation of the forgery. This allows users to easily interpret where the tampering occurred, making the model more transparent and user-friendly.

## Motivation

The motivation behind this project is to address the increasing concerns around the authenticity of digital media, especially in areas where image integrity is critical. Image forgeries are used in various malicious ways, such as spreading misinformation, committing fraud, and tampering with evidence. As tampering techniques become more advanced, traditional methods of image verification, which rely on human inspection or basic pixel-based analysis, are no longer sufficient. This project aims to provide a scalable, automated solution that can detect even the most sophisticated tampering methods by examining the fundamental properties of the image using SVD.

## Why SVD for Image Forgery Detection?

Singular Value Decomposition is a powerful mathematical tool that decomposes an image matrix into singular values, which are closely tied to the structural properties of the image. In the context of image forensics, SVD is particularly useful because tampering can alter these singular values in subtle yet detectable ways. Since SVD is sensitive to even small changes in the image's structure, it serves as an excellent tool for identifying inconsistencies in the image’s singular value patterns.

Unlike pixel-based methods, which focus on direct pixel differences between authentic and tampered images, SVD captures higher-level structural characteristics that are harder to manipulate without leaving detectable traces. This makes it particularly useful for identifying manipulations such as splicing, copy-paste, and other forms of tampering that leave structural artifacts.

## Approach Summary

1. **Image Preprocessing**: The images are resized to a standard resolution (e.g., 256x256) and converted to grayscale for simplicity. These steps standardize the input data, making it easier to apply SVD.

2. **SVD Decomposition**: Each image is decomposed using SVD, which breaks the image into three matrices: the left singular vectors, the singular values, and the right singular vectors. The singular values are then extracted for further analysis.

3. **Feature Extraction**: Statistical features are extracted from the singular values, including:
   - **Mean**: The average of the singular values.
   - **Variance**: The variability in the singular values.
   - **Energy Ratio**: The proportion of energy (information) retained by the top singular values.
   - These features are used to represent the image in a more compact form and highlight any unusual patterns indicative of tampering.

4. **Tampered Region Detection**: By applying block-based analysis, each image is divided into small regions, and the singular values for each block are analyzed. Deviations from expected values, such as sudden changes in variance or energy ratios, are flagged as potential signs of tampering.

5. **Classification**: A Convolutional Neural Network (CNN) is trained to classify images based on the extracted features. The CNN learns the relationship between the statistical features and the likelihood of an image being tampered with, allowing it to predict whether a given image is authentic or forged.

6. **Visualization**: Heatmaps are generated to visually indicate the regions of the image that show signs of tampering. This helps to not only detect tampering but also to provide a clear visualization of the regions that were altered.

## Dataset

The dataset used for this project is the [CASIA 2.0 Image Forgery Dataset](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset). This dataset contains both authentic and tampered images, allowing the model to learn the differences between genuine and manipulated content. It includes a variety of tampering techniques, including copy-paste, splicing, and other forms of image manipulation.

## Results and Conclusion

The model successfully detects image forgery by analyzing changes in the singular values using SVD. The classifier is able to distinguish between authentic and tampered images with a high degree of accuracy, and the heatmap visualizations make it easy to interpret the results. The project demonstrates the potential of SVD as a robust tool for image forensics, offering an efficient and interpretable method for forgery detection.

## Technologies Used

- **Python**: The primary programming language for developing the project.
- **NumPy**: For numerical operations and matrix manipulations, especially in SVD.
- **OpenCV**: For image processing tasks such as reading, resizing, and converting images.
- **PyTorch**: A deep learning framework used to train the CNN model for forgery classification.
- **Matplotlib**: For visualizing the results, including heatmaps to highlight tampered areas.
- **Scikit-image**: For image block division (view_as_blocks function).
- **SciPy**: Used for statistical analysis, particularly the entropy function for anomaly detection.

---

## Requirements

To run this project, you need to install the following Python libraries. You can install them using `pip`:

- numpy==1.23.4 
- opencv-python==4.6.0.66 
- scikit-image==0.19.3 
- scipy==1.9.3 
- torch==1.13.0 
- torchvision==0.14.0 
- matplotlib==3.6.2


Save these dependencies in a `requirements.txt` file for easy installation using:

```
pip install -r requirements.txt
```