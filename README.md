Forest Fire Detection using CNN
This project presents a deep learning-based image classification system designed to detect forest fires at an early stage. The system uses a Convolutional Neural Network (CNN) to classify images into two categories: fire and no fire. Early detection of forest fires is critical in minimizing damage, protecting the environment, and enabling timely emergency response.

Objective:
The main objective of this project is to develop a robust and accurate image classification model that can identify fire from images under varied lighting and background conditions. The model can serve as a foundation for integration into real-time monitoring and safety systems.

Features:
Detects the presence of fire in an image with high accuracy

Trained using a labeled dataset of fire and no fire images

Data augmentation applied for improved generalization

Designed to handle different environmental and lighting conditions

Dataset Description:
The dataset used for training consists of labeled images classified into fire and no fire categories. These images were collected from publicly available sources and prepared for deep learning tasks.

Data Preprocessing:
All images were resized to a fixed resolution to maintain consistency during training. Pixel values were normalized to a 0–1 range for faster convergence. Data augmentation techniques such as rotation, flipping, and brightness adjustments were applied to improve robustness and prevent overfitting.

Model Architecture:
The CNN model consists of convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification. Dropout layers were added to reduce overfitting. The ReLU activation function was used in hidden layers, and Softmax was used in the output layer for binary classification.

Training Process:
The model was trained in a GPU-enabled environment to improve processing speed. Accuracy and loss metrics for both training and validation sets were monitored. Hyperparameters such as learning rate, batch size, and number of epochs were adjusted to achieve the best performance.

Evaluation and Results:
The trained model was tested on a separate test dataset to evaluate its accuracy and reliability. It achieved high accuracy and was able to detect fire in various real-world scenarios with different backgrounds and lighting conditions.

Technologies Used:
Python

NumPy

Pandas

Matplotlib

Installation and Usage
Clone this repository to your local machine.

Install the required dependencies using pip install -r requirements.txt.

Run the training script to train the model on your dataset.

Use the testing script to evaluate the model on new images.

Future Improvements:
Integration with real-time video feeds for live detection

Deployment on edge devices for remote forest monitoring

Addition of smoke detection for earlier fire warnings

Conclusion
This project demonstrates the application of deep learning in environmental safety. The developed CNN model provides accurate image classification for forest fire detection, offering potential for integration into larger safety and disaster prevention systems.
