# **Pocket-Doc: A Personal Health Diagnosis Application**

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

## **Introduction**

**Pocket-Doc** is an AI-powered personal health diagnosis assistant designed to provide preliminary medical diagnoses based on user-reported symptoms. Leveraging a deep learning model trained on a diverse dataset of medical conditions, Pocket-Doc offers an accessible tool for individuals, especially those with limited access to healthcare, to receive initial assessments of their health concerns. 

This app is not a replacement for professional medical advice but serves as a supplementary tool that can guide users in seeking appropriate care.

## **Features**

- **Symptom Input Interface**: Users can input symptoms from a predefined list of unique symptoms.
- **Instant Diagnosis**: The app provides an instant diagnosis based on the input symptoms.
- **Severity Visualization**: A dynamic chart displays the severity of the diagnosis.
- **Interactive UI**: The app's background color changes based on the diagnosis outcome to provide visual feedback.
- **Detailed Descriptions**: Clicking on a diagnosis reveals a detailed description of the condition.
- **Responsive Design**: The app is fully responsive and can be used on various devices.

## **Installation**

### **Prerequisites**
- Python 3.7 or higher
- Flask
- TensorFlow
- Scikit-Learn
- JSON for storing diagnosis descriptions

### **Setup**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ashertee/pocket-doc_app/
   cd pocket-doc
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   flask --app app run
   ```
   The app will be available at `http://127.0.0.1:5000`.

## **Usage**

1. **Launch the Application**:
   Open your web browser and navigate to `http://127.0.0.1:5000`.

2. **Input Symptoms**:
   Select the symptoms you are experiencing from the list provided. You can add or remove symptoms as needed.

3. **Get a Diagnosis**:
   Click the "Get Diagnosis" button to receive an initial diagnosis. The result will be displayed along with a severity chart and a dynamically changing background color.

4. **View Diagnosis Description**:
   Click on the diagnosis to view a detailed description of the condition. The description will be displayed below the diagnosis.

5. **Review and Take Action**:
   Use the information provided to make informed decisions about seeking further medical advice or treatment.

## **Model Architecture**

The deep learning model used in Pocket-Doc is a fully connected neural network trained to classify medical conditions based on a binary input of symptoms. 

The model was trained using cross-entropy loss and optimized using the Adam optimizer.

## **Evaluation**

Pocket-Doc has undergone rigorous evaluation focusing on usability, accessibility, diagnostic accuracy, and reliability. The model was tested on a labeled dataset of 10,000 samples, achieving an accuracy of 89.2%. Expert reviews and cross-validation further supported the model’s reliability.

### **Key Results**:
- **Accuracy**: 89.2%
- **Precision**: 86.7%
- **Recall**: 84.5%
- **ROC AUC**: 0.89

The app has been tested for accessibility using tools such as WAVE and Axe, with improvements made to ensure it meets WCAG 2.1 standards.

## **Future Work**

Pocket-Doc aims to become a globally accessible medical tool, and several future developments are planned:

- **Expansion of Training Data**: Incorporating more diverse datasets, particularly for rare conditions.
- **Multi-modal Data Integration**: Enhancing diagnostic capabilities by integrating imaging and genetic data.
- **Localization**: Translating the app into multiple languages and adapting it for different cultural contexts.
- **Telemedicine Integration**: Partnering with telemedicine providers to offer direct access to healthcare professionals.



## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## **Contact**

For questions, issues, or suggestions, please contact:

- **Email**: joymaxasher@gmail.com
- **GitHub**: [https://github.com/ashertee/pocket-doc_app/](https://github.com/ashertee/pocket-doc_app/)

Pocket-Doc is constantly evolving, and your feedback is invaluable in improving the app. Thank you for using Pocket-Doc!