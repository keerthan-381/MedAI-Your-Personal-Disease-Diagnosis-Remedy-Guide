# MedAI-Your-Personal-Disease-Diagnosis-Remedy-Guide

This project is a deep learning-based disease prediction system that uses a convolutional neural network (CNN) to predict diseases based on a user's symptoms. Additionally, it suggests remedies for the predicted disease using a language model API. The system can process a user's symptom input, predict the most likely disease, and suggest remedies. 

The system uses a pre-trained CNN model, the **`cnn_prognosis_model.h5`** file, and a large language model (LLM) API to extract symptoms and suggest remedies.

## Project Structure

```
disease-prediction/
├── app.py                 # Main application file for Streamlit UI
├── prediction.py          # Script to handle disease prediction and remedies
├── training.py            # Script to train the CNN model
├── model/
│   └── cnn_prognosis_model.h5  # Trained CNN model for disease prediction
├── Data/
│   └── Training.csv       # Training data for model training and label encoding
│   └── Testing.csv        # Testing data for model evaluation
└── requirements.txt       # Project dependencies
```

### Overview of Files

1. **app.py**:
   - A Streamlit web app that allows users to input their symptoms and get disease predictions and remedies. It uses functions from `prediction.py` to extract symptoms from user input, preprocess them, and make predictions.
   
2. **prediction.py**:
   - Contains functions for extracting symptoms from user input using a large language model (LLM), preprocessing the symptoms to fit the CNN model's expected input format, making disease predictions using the CNN model, and fetching remedies using the LLM API.

3. **training.py**:
   - The script used to train the CNN model for disease prediction. It processes the training data, trains a convolutional neural network, and saves the trained model as `cnn_prognosis_model.h5`.

4. **model/cnn_prognosis_model.h5**:
   - The trained CNN model that is used for disease prediction. This model is saved in the `model` directory after training.

5. **Data/Training.csv** and **Data/Testing.csv**:
   - These CSV files contain the training and testing data respectively, which includes symptoms and their corresponding disease labels. `Training.csv` is used to train the model and perform label encoding, while `Testing.csv` is used to evaluate the model.

6. **requirements.txt**:
   - Contains the list of Python dependencies required to run the project, including packages like `tensorflow`, `streamlit`, `requests`, etc.

---

## Requirements

To run this project, you'll need the following Python dependencies:

- TensorFlow
- Streamlit
- Requests
- scikit-learn
- pandas
- numpy

To install these dependencies, use the following command:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Set Up Your Environment

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/keerthan-381/MedAI-Your-Personal-Disease-Diagnosis-Remedy-Guide.git
   cd MedAI-Your-Personal-Disease-Diagnosis-Remedy-Guide
   ```

2. Install the required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Train the CNN Model

If you don't already have the trained model (`cnn_prognosis_model.h5`), you can train the model by running `training.py`. This script will process the data, train the CNN model, and save it to the `model/` directory:

```bash
python training.py
```

This step may take some time depending on the dataset and system specifications.

### Step 3: Run the Streamlit Web App

Once the model is trained and saved, you can start the Streamlit app to interact with the disease prediction system:

```bash
streamlit run app.py
```

The Streamlit web app will launch in your default web browser, where you can input your symptoms and get disease predictions and remedies.

### Step 4: Predict Disease and Remedies

- Enter a description of your symptoms in the text area on the Streamlit web app.
- Click the "Predict Disease" button.
- The system will extract the symptoms from the input, preprocess them, and use the trained CNN model to predict the disease.
- The web app will display the predicted disease along with suggested remedies fetched from the LLM API.

---

## Model Details

The model used in this project is a **Convolutional Neural Network (CNN)**, trained using the following layers:

- **Conv1D** layers for feature extraction from symptom data.
- **Flatten** layer to convert the feature maps into a flat array.
- **Dense** layers for classification.
- **Dropout** layer for regularization.

The model is trained using categorical cross-entropy loss and accuracy metrics. It outputs the predicted disease, which is mapped back to the disease labels using label encoding.

---

## API Integration for Remedies

The **Remedies for Disease** are fetched using a large language model (LLM) API. After predicting the disease, the system makes a request to the LLM API, which returns remedies for the identified disease. These remedies are displayed in the web app.

---
## Limitations
Predictions are based on training data and should not replace professional medical advice

Accuracy depends on the quality of symptom description

Internet connection required for LLM API calls

---

## Acknowledgments

- The training data used in this project is publicly available, but ensure to check the dataset source for accuracy and relevance.
- The LLM API is used to extract symptoms and fetch remedies based on the predicted disease.

---

### Troubleshooting

- If you encounter issues with the LLM API, check your LLM token for validity.
- Ensure your environment has the necessary dependencies installed.
- Check the console logs for any errors and debug the issue based on the error messages.

### Disclaimer
This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
