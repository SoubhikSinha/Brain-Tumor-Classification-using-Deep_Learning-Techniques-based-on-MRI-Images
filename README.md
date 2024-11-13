# Brain Tumor Classification using Deep Learning Techniques based on MRI Images

<br>

### Introduction
---
Brain tumor diagnosis is a critical challenge in medical imaging, where timely and accurate detection plays a vital role in improving patient outcomes. This project, ***Brain Tumor Classification Using Deep Learning Techniques Based on MRI Images,*** leverages advancements in deep learning to enhance diagnostic accuracy and efficiency. By classifying tumors into distinct types using MRI scans, the project offers a non-invasive, swift, and reliable approach to tumor detection, aiming to assist clinicians in early diagnosis and treatment planning.

Deep learning, particularly convolutional neural networks (CNNs), has revolutionized image classification tasks due to its capability to extract complex patterns and features from visual data. This project explores multiple neural network architectures, including ***VGGNet, ResNet, DenseNet, and GoogLeNet,*** to evaluate their effectiveness in classifying brain tumors of varying grades and types. Comprehensive preprocessing and augmentation techniques are employed to ensure robust and generalized model training.

The overarching goal is to develop an accurate and efficient classification model that can serve as a proof-of-concept for real-world clinical applications. By integrating the best-performing model into an API, this project demonstrates the potential of AI-driven medical imaging solutions in reducing diagnostic errors, lowering healthcare costs, and improving patient care.

<br>
<br>

### Problem Statement
---
Brain tumors are among the most complex and life-threatening medical conditions, requiring precise diagnosis and timely treatment to improve patient outcomes. Traditional diagnostic methods often rely on manual examination of MRI scans by radiologists, which can be time-consuming, prone to human error, and limited by expertise availability. Additionally, the variability in tumor shapes, sizes, and textures presents significant challenges for accurate classification.

The primary challenges addressed in this project include 🔽

1.  **Improving Diagnostic Accuracy** ▶️ Ensuring accurate classification of brain tumors into their respective types (e.g., glioma, meningioma, pituitary) to support precise treatment planning.
2.  **Automating the Diagnostic Process** ▶️ Developing a reliable automated system that reduces dependency on manual analysis, accelerating the diagnostic workflow.
3.  **Handling Diverse Data** ▶️ Addressing the heterogeneity of MRI datasets, including differences in imaging conditions, tumor characteristics, and data imbalances.
4.  **Mitigating Overfitting** ▶️ Enhancing model generalization to perform well on unseen data while avoiding overfitting during training.
5.  **Improving Accessibility and Efficiency** ▶️ Creating a scalable, deployable solution that can be integrated into clinical workflows, especially in resource-limited settings.
<br>
By addressing these challenges, this project aims to transform brain tumor diagnosis with a deep learning-based approach, ensuring quicker, more accurate, and cost-effective medical imaging solutions for improved patient care.

<br>
<br>
<br>

### Dataset
---
#### Source of the Dataset 🔻

The dataset used in this project is publicly available on platforms such as **[Kaggle](https://www.kaggle.com)** and **[The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)**. These sources provide high-quality MRI images for brain tumor research and classification.

#### Description of the Dataset 🔻

-   **Number of Samples** ▶️ The dataset [[LINK](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-datase)] consists of more than **7000** MRI images divided into four classes ⬇️
    
    1.  **Glioma Tumor**
    2.  **Meningioma Tumor**
    3.  **Pituitary Tumor**
    4.  **Normal (No Tumor)**
 
<br>
<br>

### Data Preprocessing and Visualization
---
#### **Data Preprocessing** 🔽

The initial phase of the project involved preparing the MRI image dataset for brain tumor classification through a series of preprocessing steps:

1.  **Extraction of Dataset** 🔻
    
    -   The dataset, stored in a compressed format, was extracted to the directory : `/content/drive/MyDrive/Datasets/Brain_tumour_Kaggle` using Python’s `zipfile` module.
2.  **Organization of Dataset** 🔻
    
    -   Extracted images were segregated into training and testing sets.
    -   Further organization was done based on tumor types : **glioma, meningioma, no tumor, and pituitary**.
3.  **Loading Images** 🔻
    
    -   TensorFlow's `image_dataset_from_directory` function was employed to load images directly from the structured directory.
    -   Images were resized to **(224, 224)** pixels to standardize input dimensions.
    -   A batch size of **32** was selected to ensure efficient processing.
4.  **Class Labeling** 🔻
    
    -   Labels were automatically assigned to images based on directory names, allowing for streamlined analysis and evaluation.
5.  **Data Balancing** 🔻
    
    -   To address class imbalance, categories with excess images were downsampled to achieve uniform representation.
    -   Target counts were set at **1,321 images per class** for training and **300 images per class** for testing.

These steps ensured the dataset was well-structured, balanced, and ready for model development.

<br>

#### **Visualization Techniques** 🔽

1.  **Sample Image Visualization** 🔻
    
    -   Sample images from each class were visualized to provide insights into the dataset’s characteristics and variability.
    -   This helped in qualitatively understanding differences in tumor types and identifying potential challenges in model training.
2.  **Dataset Distribution Analysis** 🔻
    
    -   **Bar Graphs** ▶️ Illustrated the number of images in training and testing datasets across each class.
    -   **Pie Charts** ▶️ Showed the proportional representation of classes within the dataset.
3.  **Image Properties Analysis** 🔻
    
    -   **Image Size Distribution** ▶️ Analysis confirmed homogeneity in image dimensions, eliminating the need for complex transformations.
    -   **Aspect Ratio Distribution** ▶️ Uniformity in aspect ratios indicated no requirement for random cropping or rotation.

<br>

#### **Image Preprocessing Techniques** 🔽

1.  **Grayscale Conversion** 🔻
    
    -   Images were converted to grayscale using OpenCV’s `cv2` to simplify the representation and reduce computational overhead.
2.  **Image Resizing** 🔻
    
    -   Standardized dimensions of **(224, 224)** pixels were applied to ensure consistency and compatibility.
3.  **Image Normalization** 🔻
    
    -   Pixel intensities were normalized to a range of **[0, 1]**, enhancing training stability and model convergence.
4.  **Histogram Equalization** 🔻
    
    -   Applied to improve image contrast and enhance feature visibility. This technique redistributed pixel intensity values for better uniformity.
5.  **Statistical Analysis** 🔻
    
    -   Key metrics were calculated to understand pixel intensity distributions:
        -   **Mean Pixel Intensity** ▶️ ~18.47 for training and ~20.84 for testing, indicating low overall brightness.
        -   **Standard Deviation** ▶️ ~69.77 for training and ~69.56 for testing, reflecting significant intensity variability.
        -   **Intensity Range** ▶️ Min = 0, Max = 255, showcasing high-contrast regions possibly representing tumors.
<br>
These preprocessing techniques ensured the dataset’s quality and readiness for modeling tasks.
