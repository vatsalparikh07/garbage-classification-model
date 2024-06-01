# Project: AI-powered Waste Management System

## Aim: Develop an AI-powered system to improve waste management by accurately identifying and sorting different types of waste material.

## Our Solution
We aim to develop an intelligent waste classification system that can accurately identify different waste categories, such as cardboard, glass, metal, paper, plastic, and trash, using advanced computer vision and deep learning techniques.

## Workflow of the Waste Management AI System:
![image](https://github.com/vatsalparikh07/garbage-classification-model/assets/65659649/8f318baf-d16c-4e9e-98c6-5093a5bd0ab3)

1. **Object Input:** The process begins with an input image or video of waste, which must be in JPEG, JPG, PNG, or MP4 format.

2. **Inception Model:** This image is then processed by the Inception Model, which is a type of deep learning model used for image recognition.

3. **Image Processing:** The model performs image processing to prepare the image for feature extraction.

4. **Feature Extraction:** The Inception Model extracts features from the image, which are essential for recognizing the type of waste.

5. **Evaluation:** The extracted features are evaluated by the model to predict the category of waste.

6. **Predicted Category:** The model predicts the category of the waste, which could be paper, plastic, cardboard, metal, or trash.

7. **Waste Segregation and Disposal Methods:** Based on the predicted category, appropriate waste segregation and disposal methods are determined.

## Key Features
1. **Automated Sorting Process:**
   - Utilized InceptioNetV3 Machine learning model for image classification to sort different categories of waste (plastic, metal, glass, paper, and cardboard).
   - Once the waste material is identified by the model, it is automatically sorted into the appropriate category or bin, where for instance, blue dustbin would be used for recyclable material such as paper.
   
2. **Real-Time Video Processing:**
   - In addition to image classification, our AI system is capable of performing real-time video processing, enabling seamless integration into existing waste management workflows.
   - By leveraging the computational power of hardware and optimized algorithms, our model can process videos in real-time, detecting and classifying waste materials as they appear.
   
3. **Waste Disposal Guidance:**
   - When a user submits an image or video, our system not only predicts the waste category but also offers specific guidance on the appropriate disposal method and the corresponding bin or container for that particular type of waste.
   - This feature provides users with the knowledge necessary to contribute to effective waste segregation and recycling efforts, promoting environmental responsibility and sustainable practices.

![image](https://github.com/vatsalparikh07/garbage-classification-model/assets/65659649/fa60c6cd-66e9-4150-84ca-b49edfb69ca9)

Fig: A Preview of Waste Management AI System’s Website

## Development Process
Waste Management System Model Pipeline -

![image](https://github.com/vatsalparikh07/garbage-classification-model/assets/65659649/7b8c3752-4292-455a-8297-fced4e9563a2)

1. **Data Collection:** We collected an extensive dataset of waste images across various categories like plastic, metal, glass, paper, and cardboard. Images were sourced from publicly available datasets and captured photos to ensure diversity.

2. **Data Preprocessing:** The dataset underwent essential tasks like resizing and formatting to meet the input requirements of our model architecture. We split the dataset into training and testing sets with an 80-20 ratio.

3. **Data Augmentation:** Several data augmentation techniques were employed, including horizontal flipping, width/height shifting, and region filling modes, to enhance the model's learning capability.

4. **Model Training:** We trained the InceptionV3 model, a powerful convolutional neural network architecture, using transfer learning techniques. Pre-trained weights of InceptionV3 were fine-tuned on our waste dataset.

5. **Parameter Optimization:** During the training process, various parameters were fine-tuned, including activation functions (ReLU for hidden layers, softmax for output), and optimization using the Adam optimizer.

6. **Web Application Integration:** After fine-tuning the model, we integrated it into a user-friendly web application using the Streamlit framework. This application allows users to upload waste images or video streams for classification and disposal guidance.

7. **Continuous Evaluation:** Throughout the development process, we constantly evaluated the model's performance, fine-tuning hyperparameters, and adjusting the training process as needed to achieve optimal accuracy and ensure reliable waste classification results.

## Challenges
- **Ambiguous Object Identification:** Addressing instances where the system misidentifies certain materials, such as confusing plastic with glass.
- **Image Capturing through Camera Issues:** Overcoming challenges in capturing objects through the user’s camera and classifying the waste category.
- **Computational Expense in Training:** Dealing with the computational intensity and time-consuming nature of the training process.
