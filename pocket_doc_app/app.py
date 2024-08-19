from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import pickle
import os


import plotly.graph_objs as go
import plotly.io as pio

from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)
import json

# Load the model
model_path = os.path.join('model', 'pocket_doc_model.keras')
model = tf.keras.models.load_model(model_path)


######
print(model_path)

# Severity mapping (Example: You can define severity levels as per your use case)
severity_map = {
    "Allergy": 0,  # Mild
    "Varicose veins": 0,  # Mild
    "GERD": 0,  # Mild
    "Hypothyroidism": 0,  # Mild
    "Osteoarthristis": 0,  # Mild
    "Arthritis": 0,  # Mild
    "(vertigo) Paroymsal  Positional Vertigo": 0,  # Mild
    "Acne": 0,  # Mild
    "Drug Reaction": 0,  # Mild
    "Diabetes": 0,  # Mild
    "Gastroenteritis": 0,  # Mild
    "Hypertension": 0,  # Mild
    "Migraine": 0,  # Mild
    "Typhoid": 0,  # Mild
    "Psoriasis": 0,  # Mild
    "Impetigo": 0,  # Mild
    "Hepatitis A": 0,  # Mild
    "Hepatitis E": 0,  # Mild
    "Alcoholic hepatitis": 0,  # Mild
    "Common Cold": 0,  # Mild
    "Dimorphic hemmorhoids(piles)": 0,  # Mild
    "Fungal infection": 4,  # Moderate
    "Peptic ulcer diseae": 4,  # Moderate
    "Chronic cholestasis": 4,  # Moderate
    "Bronchial Asthma": 4,  # Moderate
    "Cervical spondylosis": 4,  # Moderate
    "Malaria": 4,  # Moderate
    "Dengue": 4,  # Moderate
    "Hypoglycemia": 4,  # Moderate
    "Urinary tract infection": 4,  # Moderate
    "Hepatitis B": 4,  # Moderate
    "Hepatitis C": 4,  # Moderate
    "Hepatitis D": 4,  # Moderate
    "AIDS": 7,  # Severe
    "Jaundice": 7,  # Severe
    "Chicken pox": 7,  # Severe
    "Tuberculosis": 7,  # Severe
    "Heart attack": 9,  # Critical
    "Paralysis (brain hemorrhage)": 9,  # Critical
}
#######

# Initialize Flask app
app = Flask(__name__)

# Load label encoder
label_encoder_path = os.path.join('model', 'label_encoder.pkl')
with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

# List of all symptoms (make sure this matches the columns in your training data)
all_symptoms = [
        'Itching',
        'Skin Rash',
        'Nodal Skin Eruptions',
        'Continuous Sneezing',
        'Shivering',
        'Chills',
        'Joint Pain',
        'Stomach Pain',
        'Acidity',
        'Ulcers On Tongue',
        'Muscle Wasting',
        'Vomiting',
        'Burning Micturition',
        'Spotting Urination',
        'Fatigue',
        'Weight Gain',
        'Anxiety',
        'Cold Hands and Feets',
        'Mood Swings',
        'Weight Loss',
        'Restlessness',
        'Lethargy',
        'Patches in Throat',
        'Irregular Sugar Level',
        'Cough',
        'High Fever',
        'Sunken Eyes',
        'Breathlessness',
        'Sweating',
        'Dehydration',
        'Indigestion',
        'Headache',
        'Yellowish Skin',
        'Dark Urine',
        'Nausea',
        'Loss of Appetite',
        'Pain Behind the Eyes',
        'Back Pain',
        'Constipation',
        'Abdominal Pain',
        'Diarrhoea',
        'Mild Fever',
        'Yellow Urine',
        'Yellowing of Eyes',
        'Acute Liver Failure',
        'Fluid Overload',
        'Swelling of Stomach',
        'Swelled Lymph Nodes',
        'Malaise',
        'Blurred and Distorted Vision',
        'Phlegm',
        'Throat Irritation',
        'Redness of Eyes',
        'Sinus Pressure',
        'Runny Nose',
        'Congestion',
        'Chest Pain',
        'Weakness in Limbs',
        'Fast Heart Rate',
        'Pain during Bowel Movements',
        'Pain in Anal Region',
        'Bloody Stool',
        'Irritation in_Anus',
        'Neck Pain',
        'Dizziness',
        'Cramps',
        'Bruising',
        'Obesity',
        'Swollen Legs',
        'Swollen Blood Vessels',
        'Puffy Face and Eyes',
        'Enlarged Thyroid',
        'Brittle Nails',
        'Swollen Extremities',
        'Excessive Hunger',
        'Extra Marital Contacts',
        'drying and Tingling Lips',
        'Slurred Speech',
        'Knee Pain',
        'Hip Joint Pain',
        'Muscle Weakness',
        'Stiff Neck',
        'Swelling Joints',
        'Movement Stiffness',
        'Spinning Movements',
        'Loss of Balance',
        'Unsteadiness',
        'Weakness of One Body Side',
        'Loss of Smell',
        'Bladder Discomfort',
        'Foul Smell of Urine',
        'Continuous Feel of Urine',
        'Passage of Gases',
        'Internal Itching',
        'Toxic Look (Typhos)',
        'Depression',
        'Irritability',
        'Muscle Pain',
        'Altered Sensorium',
        'Red Spots over Body',
        'Belly Pain',
        'Abnormal Menstruation',
        'Dischromic Patches',
        'Watering from Eyes',
        'Increased Appetite',
        'Polyuria',
        'Family History',
        'Mucoid Sputum',
        'Rusty Sputum',
        'Lack of Concentration',
        'Visual Disturbances',
        'Receiving Blood Transfusion',
        'Receiving Unsterile Injections',
        'Coma',
        'Stomach Bleeding',
        'Distention of Abdomen',
        'History of Alcohol Consumption',
        'Fluid Overload',
        'Blood in Sputum',
        'Prominent Veins on Calf',
        'Palpitations',
        'Painful Walking',
        'Pus Filled Pimples',
        'Blackheads',
        'Scurring',
        'Skin Peeling',
        'Silver like Rusting',
        'Small Dents in Nails',
        'Inflammatory Nails',
        'Blister',
        'red sore Around Nose',
        'Yellow Crust Ooze',

]


def preprocess_input(selected_symptoms):
    # Create a zero vector with the same length as the number of features (304)
    input_vector = np.zeros(len(all_symptoms), dtype=int)
    print((input_vector))

    # Set the positions of the selected symptoms to 1
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            input_vector[index] = 1

    # from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    # encoder = OneHotEncoder()
    # encoded_input = encoder.fit_transform(input_vector.reshape(1, -1)).toarray()
    # print( encoded_input.shape)
    #
    # #####
    # import pandas as pd
    # one_hot_df = pd.DataFrame(0, index=[0], columns=all_symptoms)
    #
    # # Update the DataFrame: for each symptom in the input, set the corresponding column to 1
    # for symptom in selected_symptoms:
    #         if symptom in one_hot_df.columns:
    #                 one_hot_df.at[0, symptom] = 1
    #
    # # Display the resulting DataFrame
    # print(one_hot_df)
    # print(one_hot_df.describe())
    # # X = one_hot_df.iloc[:, :]
    # encoder = OneHotEncoder()
    # encoded_input = encoder.fit_transform(one_hot_df).toarray()
    # print( encoded_input.shape)
    # #######


    return input_vector.reshape(1, -1)  # Reshape for the model


@app.route('/')
def index():

    with open('static/diagnosis_description.json') as f:
        descriptions = json.load(f)

    return render_template('index.html', symptoms=all_symptoms, descriptions=descriptions)


# #########
# @app.route('/')
# def index():
#     with open('static/diagnosis_descriptions.json') as f:
#         descriptions = json.load(f)
#     return render_template('index.html', descriptions=descriptions)
# #############


@app.route('/diagnose', methods=['POST'])
def diagnose():
    selected_symptoms = request.json['selected_symptoms']
    if len(selected_symptoms) < 2:
        return jsonify({'error': 'Please select at least two symptoms for a diagnosis.'}), 400

    input_vector = preprocess_input(selected_symptoms)

    # with open('pr.txt', 'r') as file:
    #
    #         # Pass the input vector directly to the model
    #         print(file.read())
    #         # prediction = model.predict(file.read())

    print(input_vector)
    print(input_vector.shape)
    prediction = model.predict(input_vector)
    print(prediction)

    diagnosis = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    ##########
    predicted_class = np.argmax(prediction, axis=1)

    # Ensure the predicted class is within the known labels
    if predicted_class[0] >= len(label_encoder.classes_):
            return jsonify({'error': 'Model predicted an unknown label.'}), 500

    # Get the severity level
    severity = severity_map.get(diagnosis, 5)  # Default to moderate if not found

    # Generate the plot
    plot_div = create_severity_plot(diagnosis, severity)

    print(f'test  ', severity)
    print(f'test  ', severity_map)
    # print(f'test  ', plot_div)





    # Load the diagnosis descriptions from the JSON file
    with open('static/diagnosis_description.json') as f:
        descriptions = json.load(f)

    # Get the corresponding description from the JSON file
    description = descriptions.get(diagnosis, "No description available for this diagnosis.")

    print(diagnosis)
    print(description)
    print(descriptions)






    # Emit data to the client for dynamic plot update
    socketio.emit('update_plot', {'diagnosis': diagnosis, 'severity': severity})

    return jsonify({'diagnosis': diagnosis, 'description': description})




    ############


    print(f'Diagnosis ~Cast: ', label_encoder.inverse_transform([0]))

    return jsonify({'diagnosis': diagnosis})


def create_severity_plot(diagnosis, severity):
        # Create a simple bar chart to represent severity
        severity_levels = ["Mild", "Moderate", "Severe", "Critical"]
        values = [0, 4, 7, 9]

        trace = go.Bar(x=severity_levels, y=values, name='Severity Levels')
        trace_diagnosis = go.Bar(x=[severity_levels[(severity - 1) // 2]], y=[severity], name=diagnosis,
                                 marker_color='red')

        layout = go.Layout(
                title=f"Severity of Diagnosis: {diagnosis}",
                xaxis={'title': 'Severity Level'},
                yaxis={'title': 'Severity Score'},
                barmode='group'
        )

        fig = go.Figure(data=[trace, trace_diagnosis], layout=layout)

        # Convert plotly figure to HTML div string
        plot_div = pio.to_html(fig, full_html=False)

        return plot_div


#######

@app.route('/get_description/<diagnosis>')
def get_description(diagnosis):
    with open('static/diagnosis_description.json') as f:
        descriptions = json.load(f)
    description = descriptions.get(diagnosis, "No description available for this diagnosis.")
    return jsonify({'description': description})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
