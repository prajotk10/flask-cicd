from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd  # Ensure you're using pandas correctly

app = Flask(__name__)

# Global variable to hold the loaded model and treatment data
model = None
treatment_data = None  # Initialize the treatment_data variable

def load_model():
    global model
    if model is None:
        model = pickle.load(open(r'./model/RFC.pkl', 'rb'))  # Use double backslashes for Windows paths
    return model

def load_treatment_data():
    global treatment_data
    if treatment_data is None:
        treatment_data = pd.read_csv(r'./model/treatment.csv', encoding='latin1')  # Adjust encoding if needed
    return treatment_data


def get_treatment_info(disease):
    global treatment_data
    
    # Ensure treatment_data is loaded
    if treatment_data is None:
        load_treatment_data()  # Load treatment data if it hasn't been loaded yet

    treatment_row = treatment_data[treatment_data['Disease'] == disease]

    if not treatment_row.empty:
        treatments = treatment_row[['Treatment 1', 'Treatment 2', 'Treatment 3']].values[0]
        treatments = [t for t in treatments if pd.notna(t) and t != '']
        
        if treatments:
            return ', '.join(treatments)  # Return treatments as a comma-separated string
        else:
            return "No treatment information available."
    else:
        return "No treatment information available."




# Define the model features
model_features = [
    'Blindness', 'Bronchitis', 'Canker', 'Caseation', 'Cephalitis', 'Coccidiosis',
    'Distension', 'Dyspnea', 'Inflammation', 'Tremors', 'Asphyxiation', 'Cyanosis',
    'Disheveled', 'Immobility', 'Malaise', 'Mycosis', 'Oviposition', 'Respiratory_distress',
    'Rhinorrhea', 'Arthralgia', 'Dehydration', 'Edema', 'Immobilization', 'Inactivity',
    'Kyphosis', 'Neoplasia', 'Omphalitis', 'Plucking', 'Rhinorrhea.1', 'Ulceration',
    'Epiphora', 'Listlessness', 'Opacification', 'Ovulation', 'Pasty', 'Splenomegaly',
    'Suppuration', 'Toxin', 'Dampness', 'Diarrhea', 'Stop_laying', 'Sudden_death',
    'Petechiae'
]

disease_mapping = {
    0: 'Avian Influenza',
    1: 'Botulism', 
    2: 'Bumblefoot',
    3: 'Fowl Cholera',
    4: 'Fowl Pox',
    5: 'Infectious Bronchitis',
    6: 'Infectious Coryza',
    7: 'Mareks Disease',
    8: 'Mushy Chick Disease',
    9: 'Newcastle Disease',
    10: 'Pullorum',
    11: 'Quail Disease',
    12: 'Thrush'
}

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if request.method == 'POST':
        load_treatment_data()  # Ensure treatment data is loaded at the start of this function
        # Extract symptoms from POST data
        symptoms = request.form.to_dict()  # Convert ImmutableMultiDict to a regular dict
        
        # Initialize feature vector with zeros
        feature_vector = [0] * len(model_features)
        predicted_disease = "No disease selected."  # Initialize the variable

        # Check if any symptoms are selected
        if not any(symptoms.get(symptom) == 'on' for symptom in model_features):
            return render_template('predict_disease.html', prediction="Please select at least one symptom.")

        # Populate feature vector based on symptoms present
        for symptom in model_features:
            if symptoms.get(symptom) == 'on':  # 'on' indicates the checkbox is checked
                feature_vector[model_features.index(symptom)] = 1

        # Load the machine learning model
        model = load_model()

        try:
            # Make prediction
            prediction = model.predict([feature_vector])
            predicted_disease = disease_mapping[prediction[0]]  # Use the mapping
            treatment_info = get_treatment_info(predicted_disease)
            prediction_text = f"Predicted Disease: {predicted_disease}\nTreatment: {treatment_info}"
        except Exception as e:
            # Handle any unexpected exceptions
            return jsonify({'error': str(e)}), 500  # Return a 500 Internal Server Error

        return render_template('predict_disease.html', prediction=prediction_text)

    return jsonify({'error': 'Invalid request method'}), 405  # Method Not Allowed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/basic_prevention')
def basic_prevention():
    return render_template('basic_prevention.html')

@app.route('/basic_treatment')
def basic_treatment():
    return render_template('basic_treatment.html')

@app.route('/symptom_form')
def symptom_form():
    return render_template('symptom_form.html')

@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')

@app.route('/privacy_and_cookies_policy')
def privacy_and_cookies_policy():
    return render_template('privacy_and_cookies_policy.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

@app.route('/tc')
def tc():
    return render_template('tc.html')

@app.route('/sign_up')
def sign_up():
    return render_template('sign_up.html')

@app.route('/log_in')
def log_in():
    return render_template('log_in.html')

@app.route('/home')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
