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

<br>
<br>

### Model Architecture
---
### VGGNet (VGG-13) ⬇️

#### Overview of VGG13 Model 🔻

VGG13, a variant of the VGGNet family, was selected for this project due to its balance between computational efficiency and performance. The architecture is well-suited for image classification tasks, leveraging deep convolutional layers to extract complex features from input images. The choice of VGG13 over deeper architectures like VGG16 and VGG19 was motivated by computational resource constraints while maintaining high classification accuracy.

#### Key Components of VGG13 Architecture ⬇️

1.  **Convolutional Layers**
    
    -   Comprises 13 convolutional layers grouped into blocks, with each block followed by a ReLU activation function.
    -   These layers extract hierarchical features from the input images, starting from low-level features like edges and progressing to high-level features such as tumor patterns.
2.  **Pooling Layers**
    
    -   Max pooling layers with a kernel size of 2x2 and a stride of 2 reduce the spatial dimensions of feature maps.
    -   Pooling helps retain essential information while minimizing computational complexity.
3.  **Fully Connected Layers**
    
    -   After flattening the feature maps, the architecture includes two fully connected layers for learning complex representations of the data.
    -   A final output layer with four neurons (corresponding to the classes: glioma, meningioma, no tumor, and pituitary) and a softmax activation function outputs class probabilities.
4.  **Batch Normalization**
    
    -   Batch normalization is applied after each convolutional and fully connected layer to stabilize activations, improve training speed, and enhance generalization.

<br>

#### Hyperparameter Choices ⬇️

1.  **Learning Rate**
    
    -   Initially set to 0.001, the learning rate was tuned to balance the trade-off between convergence speed and stability.
2.  **Optimizer**
    
    -   The Adam optimizer was chosen for its adaptive learning rate and robust performance in complex optimization tasks.
3.  **Loss Function**
    
    -   Cross-entropy loss was employed as the objective function, given its suitability for multi-class classification tasks.
4.  **Regularization**
    
    -   L2 regularization was implemented in later setups to mitigate overfitting by penalizing large weights.
5.  **Epochs and Batch Size**
    
    -   The model was trained over 20 epochs with a batch size of 32, providing a balance between computational efficiency and convergence.

<br>

### ResNet18 ⬇️

#### Overview of ResNet18 Model 🔻

ResNet18, a member of the ResNet family, was utilized in this project for its innovative residual learning approach. The architecture effectively mitigates the vanishing gradient problem through shortcut connections, enabling deeper network training without performance degradation. Its balance of simplicity and efficiency makes it well-suited for the classification of complex brain tumor types.

#### Key Components of ResNet18 Architecture ⬇️

1.  **Residual Blocks**
    
    -   Composed of four main residual layers, each containing multiple **BasicBlock** units.
    -   Each block uses a shortcut connection to skip certain layers, ensuring stable gradient flow during backpropagation.
    -   Blocks have varying feature planes:
        -   **Layer 1:** Two BasicBlocks with 64 feature planes (stride 1).
        -   **Layer 2:** Two BasicBlocks with 128 feature planes (stride 2).
        -   **Layer 3:** Two BasicBlocks with 256 feature planes (stride 2).
        -   **Layer 4:** Two BasicBlocks with 512 feature planes (stride 2).
2.  **Convolutional Layers**
    
    -   Initial convolution layer with a 7x7 kernel and stride 2, followed by max pooling.
    -   Convolution layers within BasicBlocks use 3x3 kernels, ReLU activations, and batch normalization.
3.  **Pooling Layers**
    
    -   Max pooling is applied after the initial convolution layer to reduce spatial dimensions while retaining important features.
    -   Global average pooling (7x7 kernel) is applied after the final residual block to produce fixed-size feature maps.
4.  **Fully Connected Layer**
    
    -   The flattened feature maps are passed through a fully connected layer to output class probabilities.
    -   The final layer consists of four neurons corresponding to the classes: glioma, meningioma, no tumor, and pituitary.
5.  **Batch Normalization**
    
    -   Applied after every convolutional layer, stabilizing activations and accelerating convergence.

<br>

#### Hyperparameter Choices ⬇️

1.  **Learning Rate**
    
    -   Initially set to **0.001** in most setups to ensure balanced convergence and stability.
    -   In **Setup 4**, the learning rate was increased to **0.01**, leading to rapid improvement but occasional instability.
2.  **Optimizer**
    
    -   The Adam optimizer was selected for its adaptive learning rate and efficient gradient handling.
3.  **Loss Function**
    
    -   Cross-entropy loss was used to measure discrepancies between predicted probabilities and true labels.
4.  **Regularization Techniques**
    
    -   L2 regularization (weight decay) was added to improve generalization by penalizing large weights.
    -   Dropout (0.1) was employed in advanced setups to reduce overfitting.
5.  **Epochs and Batch Size**
    
    -   Initial training was conducted for 10 epochs, later extended to 25, 35, and 50 epochs in different setups to evaluate performance improvements.
    -   Batch size was fixed at **32**, balancing computational efficiency and convergence.

<br>

### DenseNet ⬇️

#### Overview of DenseNet Model 🔻

DenseNet is a deep neural network architecture known for its dense connections between layers, where each layer receives input from all preceding layers. This dense connectivity allows the model to better propagate information throughout the network, leading to improved learning and reduced vanishing gradient problems. DenseNet is particularly suited for tasks like image classification, where deep feature extraction is essential. By using a growth rate parameter, the architecture can flexibly control the number of output channels in each block, making it both efficient and scalable for complex tasks like tumor classification from medical images.

#### Key Components of DenseNet Architecture ⬇️

1.  **BasicBlock**
    
    -   The fundamental building block of DenseNet, comprising a 3x3 convolutional layer, batch normalization, and ReLU activation.
    -   Optionally includes a dropout layer after ReLU for regularization.
    -   A concatenation operation (`torch.cat`) merges the output of each block with its input, creating a dense connection that allows for efficient feature propagation.
2.  **Dense Blocks**
    
    -   The DenseNet model consists of several dense blocks, each containing multiple BasicBlock layers.
    -   The growth rate controls how many new channels each block adds to the network, ensuring that the model’s complexity grows gradually.
3.  **Transition Layers**
    
    -   Transition layers are employed between dense blocks to manage the increasing number of channels. These layers apply a 1x1 convolution followed by batch normalization, ReLU activation, and average pooling to reduce dimensionality.
4.  **Final Classification Layer**
    
    -   After all dense blocks, the model includes a final batch normalization layer followed by adaptive average pooling to reduce the feature map to a 1x1 spatial size.
    -   A fully connected layer maps the features to the output classes, providing the final classification predictions.

<br>

#### Hyperparameter Choices ⬇️

1.  **Learning Rate**
    
    -   Initially set to 0.001, the learning rate was tuned to balance convergence speed and stability, especially in deeper models like DenseNet.
2.  **Optimizer**
    
    -   The Adam optimizer was selected for its adaptive learning rate, making it effective for complex tasks where the data and loss landscapes are varied.
3.  **Loss Function**
    
    -   Cross-entropy loss was chosen as the objective function, which is ideal for multi-class classification tasks, like tumor classification into different categories.
4.  **Regularization**
    
    -   L2 regularization was implemented to combat overfitting and ensure that the model generalizes better to unseen data.
5.  **Epochs and Batch Size**
    
    -   The DenseNet model was trained with different epoch settings, ranging from 10 to 50 epochs, and varied batch sizes to optimize training efficiency.

<br>

### GoogLeNet ⬇️

#### Overview of GoogLeNet Model 🔻

GoogLeNet, also known as Inception v1, is a deep convolutional neural network architecture that is designed to improve computational efficiency while achieving high accuracy on complex tasks like image classification. It introduced the **Inception module**, which enables the network to compute multiple types of convolutions with different kernel sizes in parallel within the same layer. This allows the network to capture features at various spatial scales. The architecture also introduced **global average pooling** and drastically reduced the number of parameters compared to traditional CNNs, making it highly efficient.

#### Key Components of GoogLeNet Architecture ⬇️

1. **Inception Module**
    - The core innovation of GoogLeNet is the Inception module, which applies multiple convolution filters with different kernel sizes (e.g., 1x1, 3x3, 5x5) and combines the results.
    - These filters run in parallel within the same layer, allowing the network to capture diverse features at different levels of abstraction without increasing the computational cost too much.
    - A 1x1 convolution is used for dimensionality reduction, reducing the computational load.
  
2. **Global Average Pooling (GAP)**
    - Instead of using fully connected layers at the end of the network, GoogLeNet employs global average pooling to reduce the feature map to a single value per channel.
    - This approach eliminates the need for large fully connected layers, helping to reduce the number of parameters and preventing overfitting.
  
3. **Auxiliary Classifiers**
    - GoogLeNet includes auxiliary classifiers, which are intermediate classifiers added at various layers during training. These classifiers provide additional gradient signals to help with the training of deeper layers and prevent overfitting.
    - During inference, these classifiers are not used, but they provide regularization during training.

4. **Final Classification Layer**
    - The final output layer is a softmax function applied to the output of the global average pooling layer, producing the class probabilities for classification tasks.
  
#### Hyperparameter Choices ⬇️

1. **Learning Rate**
    - The initial learning rate is typically set to 0.01, followed by a learning rate decay during training to fine-tune the model.
  
2. **Optimizer**
    - Adam optimizer is commonly used for training GoogLeNet due to its adaptive learning rates, which helps in dealing with large networks and complex datasets.
  
3. **Loss Function**
    - Cross-entropy loss is used for classification tasks, as it is suitable for multi-class classification problems.

4. **Regularization**
    - Dropout is implemented in the Inception modules and between the fully connected layers to prevent overfitting.
  
5. **Epochs and Batch Size**
    - GoogLeNet is typically trained for a large number of epochs (ranging from 50 to 100) with a moderate batch size (e.g., 32 or 64) for efficient training.
  
6. **Auxiliary Classifiers' Weight**
    - The auxiliary classifiers are weighted (often by a factor of 0.3 or 0.4) during training to balance their contribution to the overall loss.

