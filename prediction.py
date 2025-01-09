import os
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Set your LLM token directly here
LLM_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImtyaXNobmFzYWlrZWVydGhhbi5uYWdhbmRsYUBncmFtZW5lci5jb20ifQ._9tWlsYeuhpfHi0aKTZGQE3FC3PjS5igzSQ0qtG_Zbc"  # Replace with your actual token

# Load the trained model
model = load_model("model/cnn_prognosis_model.h5")

# Load the training data for label encoding
train_data = pd.read_csv("Data/Training.csv")
train_data = train_data.iloc[:, :-1]
X_train = train_data.drop("prognosis", axis=1)
y_train = train_data["prognosis"]

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)


# Function to extract symptoms from the user prompt using LLM
def extract_symptoms_from_prompt(user_prompt, feature_names):
    prompt = (
        f"Extract the symptoms from this text: '{user_prompt}'. "
        "Return the symptoms as a comma-separated list matching these features:\n"
        f"{', '.join(feature_names)}"
    )
    
    response = requests.post(
        "https://llmfoundry.straive.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {LLM_TOKEN}:my-test-project"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    llm_response = response.json()
    symptoms = llm_response['choices'][0]['message']['content']
    
    # Extract the list of symptoms (assumed to be a comma-separated string)
    symptoms_list = symptoms.split(",")
    
    # Display extracted symptoms
    print("Extracted Symptoms:", symptoms_list)
    
    return symptoms_list



# Function to preprocess symptoms for the CNN model
def preprocess_symptoms(symptoms_list, X_train):
    # Convert symptoms to the format used by the CNN model
    # Assuming the same features as in X_train
    feature_vector = np.zeros(X_train.shape[1])  # Create a zero vector with the correct number of features
    for symptom in symptoms_list:
        # Assuming symptom matches a column name or feature, set the corresponding index to 1
        # Example: map symptoms to corresponding feature columns (adjust as needed)
        if symptom in X_train.columns:
            feature_vector[X_train.columns.get_loc(symptom)] = 1
    return feature_vector.reshape(1, -1, 1)  # Reshape for Conv1D input


# # Function to get remedies for the predicted disease from LLM
def get_remedies_for_disease(disease):
    prompt = f"Provide remedies for the disease: {disease}. List common treatments or suggestions to manage this disease."

    # Sending the request to the LLM API for remedies
    response = requests.post(
        "https://llmfoundry.straive.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {LLM_TOKEN}:my-test-project"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    llm_response = response.json()
    remedies = llm_response['choices'][0]['message']['content']
    
    return remedies

# Function to predict the disease and get remedies
def predict_disease_and_get_remedies(user_prompt):
    # Define the feature names (list of symptoms)
    feature_names = [
        'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
        'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
        'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 
        'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 
        'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 
        'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
        'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 
        'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
        'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 
        'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
        'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 
        'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
        'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
        'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
        'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 
        'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases',
        'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 
        'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 
        'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 
        'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 
        'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 
        'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 
        'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 
        'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
    ]
    
    # Extract symptoms using the LLM model, passing feature_names
    symptoms_list = extract_symptoms_from_prompt(user_prompt, feature_names)
    
    # Preprocess symptoms for the CNN model
    X_new = preprocess_symptoms(symptoms_list, X_train)
    
    # Predict the disease
    y_pred_onehot = model.predict(X_new)
    y_pred = np.argmax(y_pred_onehot, axis=1)
    
    # Decode the prediction into a disease name
    disease = label_encoder.inverse_transform(y_pred)[0]
    
    # Get remedies for the disease
    remedies = get_remedies_for_disease(disease)
    
    return disease, remedies

# Example user prompt
user_prompt = "I have been experiencing fever, chills, fatigue, headache, body aches, sore throat, cough, difficulty breathing, nausea, vomiting, loss of appetite, muscle weakness, dizziness, back pain, joint pain, abdominal pain, diarrhea, constipation, and a runny nose. My skin appears pale, and I have noticed some swelling in my legs. I've also been feeling anxious and restless, and I've had trouble sleeping at night."

# Predict the disease and get remedies
predicted_disease, remedies = predict_disease_and_get_remedies(user_prompt)

# Display disease and remedies separately
print(f"Predicted Disease: {predicted_disease}")
print("\nRemedies:")
print(remedies)
