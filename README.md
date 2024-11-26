# Bone Age Prediction Using ResNet-50

## Project Overview

This project aims to predict bone age from X-ray images using a convolutional neural network (CNN) based on the ResNet-50 architecture. The goal is to accurately estimate bone age in children, assisting in diagnosing growth disorders and medical assessments.

---

## Key Features

- **Dataset**: RSNA Bone Age dataset, consisting of labeled X-ray images for training and testing.
- **Preprocessing**: Images are resized, augmented, and normalized for optimal training.
- **Model Architecture**: 
  - Pretrained ResNet-50 modified for grayscale input (single-channel images).
  - Custom fully connected layers to predict bone age.
- **Evaluation**: Performance measured by Mean Absolute Error (MAE) and loss metrics.

---

## Methodology

### 1. **Data Processing**
   - Loaded the RSNA Bone Age dataset containing training and test image paths along with labels.
   - Applied data augmentation techniques such as random rotation, flipping, and affine transformations to improve generalization.

### 2. **Dataset and Dataloader**
   - Custom `BoneAgeDataset` class for handling image loading and transformation.
   - Split data into training and test sets (80:20 split).
   - Used PyTorch `DataLoader` for efficient batch processing.

### 3. **Model Architecture**
   - Utilized a pretrained ResNet-50 model.
   - Modified the first convolution layer to accept grayscale images.
   - Replaced the fully connected layer to output a single value for bone age prediction.

### 4. **Training**
   - Loss function: Mean Squared Error (MSE).
   - Optimizer: Adam with a learning rate of 0.001.
   - Trained the model for 5 epochs with mini-batches to minimize the loss.

### 5. **Evaluation**
   - Test loss and Mean Absolute Error (MAE) calculated to evaluate the model.
   - Predictions compared with true values to assess performance.

---

## Results

| **Metric**          | **Value**  |
|----------------------|------------|
| Training Loss (Epoch 5) | 649.5476  |
| Test Loss (Epoch 5)     | 506.3794  |
| Mean Absolute Error (MAE) | 17.3709  |

---

## Tools and Libraries

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Preprocessing**: PIL (Python Imaging Library)

---

## Insights and Challenges

- **Data Augmentation**: Helped reduce overfitting and improved model generalization.
- **Gray-scale Conversion**: Adapting ResNet-50 for single-channel images was crucial for medical imaging.
- **Model Evaluation**: MAE provided an intuitive measure of the model's accuracy in estimating bone age.

---

## Future Improvements

- **Increase Training Data**: Incorporate more X-ray images for diverse training.
- **Advanced Architectures**: Experiment with state-of-the-art models like Vision Transformers.
- **Fine-tuning Hyperparameters**: Optimize learning rate, batch size, and augmentations.
- **Clinical Validation**: Test the model's predictions on unseen clinical data.

---

## Conclusion

The project successfully demonstrates the application of ResNet-50 for predicting bone age from X-ray images. Despite the limited training epochs, the model achieved a competitive MAE, showing promise for medical applications in pediatric healthcare.

--- 

## Contact

For questions or collaborations, feel free to reach out!
