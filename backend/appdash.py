import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import shap
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import random
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app with custom static and template folders
app = Flask(__name__, 
            static_folder='../frontend/static', 
            template_folder='../frontend/templates')
CORS(app)

# Load models and scalers
try:
    eng_model = pickle.load(open('xgboost_model_cv.pkl', 'rb'))
    eng_scaler = pickle.load(open('scaler_cv.pkl', 'rb'))
    app_model = pickle.load(open('xgboost_applied_base.pkl', 'rb'))
    app_scaler = pickle.load(open('scaler_applied.pkl', 'rb'))
    # Log scaler details
    logger.debug(f"eng_scaler: type={type(eng_scaler)}, shape={eng_scaler.shape if isinstance(eng_scaler, np.ndarray) else 'N/A'}, sample={eng_scaler[:5] if isinstance(eng_scaler, np.ndarray) else eng_scaler}")
    logger.debug(f"app_scaler: type={type(app_scaler)}, shape={app_scaler.shape if isinstance(app_scaler, np.ndarray) else 'N/A'}, sample={app_scaler[:5] if isinstance(app_scaler, np.ndarray) else app_scaler}")
    # Log model feature names if available
    logger.debug(f"eng_model feature_names: {getattr(eng_model, 'feature_names', 'Not available')}")
    logger.debug(f"app_model feature_names: {getattr(app_model, 'feature_names', 'Not available')}")
except FileNotFoundError as e:
    logger.error(f"Error: Model or scaler file not found: {e}")
    exit(1)

# Feature definitions
eng_features = [
    'Residence', 'FTEN Status', 'Bursary', 'Offering Type', 'Disability',
    'Age Group', 'Exam Type', 'Exam Month', 'Race', 'Gender', 'Stud_Type', 'Marital Status'
]
app_features = [
    'Department', 'Marital Status', 'Gender', 'Race', 'Disability', 'Stud Type',
    'Exam Month', 'Exam Type', 'Offer Type', 'FTEN Status', 'Final Year', 'Residence', 'Bursary', 'Age Group'
]

# Possible values for features with label encoding mappings
feature_values = {
    'Residence': {'Yes': 1, 'No': 0},
    'FTEN Status': {'N': 0, 'F': 1, 'T': 2, 'E': 3},
    'Bursary': {'Yes': 1, 'No': 0},
    'Offering Type': {'D1': 0, 'D3': 1, 'P1': 2, 'I1': 3, 'P3': 4, 'I3': 5, 'D2': 6, 'D6': 7, 'D8': 8, 'I7': 9, 'D7': 10, 'P7': 11},
    'Disability': {'Yes': 1, 'No': 0},
    'Age Group': {'15-19': 0, '20-29': 1, '30-39': 2, '40-49': 3, '>50': 4},
    'Exam Type': {'Normal': 0, 'Re-Exam': 1, 'Special': 2, 'Experiential Training': 3},
    'Exam Month': list(range(1, 13)),  # Numerical, no encoding needed
    'Race': {'African': 0, 'Indian': 1, 'White': 2, 'Black': 3, 'Non SA': 4},
    'Gender': {'Male': 0, 'Female': 1},
    'Stud_Type': {'Normal': 0, 'SADC': 1, 'International': 2, 'Refugee': 3, 'Asylum Seeker': 4},
    'Marital Status': {'Single': 0, 'Married': 1},
    'Department': {
        'Maritime Studies': 0, 'Biotechnology & Food Science': 1, 'Horticulture': 2,
        'Textile Science & App Tech': 3, 'Food & Nutrition Consumer Science': 4, 
        'Faculty Office - Applied Science': 5
    },
    'Final Year': {'Yes': 1, 'No': 0},
    'Offer Type': {'D1': 0, 'D3': 1, 'P1': 2, 'I1': 3, 'P3': 4, 'I3': 5, 'D2': 6, 'D6': 7, 'D8': 8, 'I7': 9, 'D7': 10, 'P7': 11},
    'Stud Type': {'Yes': 1, 'No': 0}  # Note: 'Stud Type' vs 'Stud_Type' naming inconsistency
}

# In-memory storage for predictions
predictions_storage = {'engineering': [], 'applied_sciences': []}

# Initialize SHAP explainers
eng_explainer = shap.TreeExplainer(eng_model)
app_explainer = shap.TreeExplainer(app_model)

def generate_student_id():
    return str(random.randint(10000000, 99999999))

def generate_random_student(faculty):
    features = eng_features if faculty == 'engineering' else app_features
    student = {'student_id': generate_student_id()}
    for feature in features:
        if feature in feature_values:
            if feature == 'Exam Month':
                student[feature] = random.choice(feature_values[feature])  # Integer
            else:
                student[feature] = random.choice(list(feature_values[feature].keys()))
        else:
            student[feature] = random.choice(['Yes', 'No'])
    logger.debug(f"Generated student: {student}")
    return student

def encode_features(student, features):
    encoded = {}
    for feature in features:
        value = student[feature]
        if feature == 'Exam Month':
            try:
                encoded[feature] = int(value)
                if encoded[feature] not in feature_values['Exam Month']:
                    raise ValueError(f"Exam Month value {encoded[feature]} not in {feature_values['Exam Month']}")
            except (ValueError, TypeError):
                logger.error(f"Invalid Exam Month value for student {student['student_id']}: {value}")
                raise ValueError(f"Exam Month must be an integer between 1 and 12, got {value}")
        else:
            try:
                encoded[feature] = feature_values[feature][value]
            except KeyError:
                logger.error(f"Invalid value for {feature}: {value}")
                raise ValueError(f"Invalid value for {feature}: {value}")
    return encoded

def get_expected_features(features):
    """Return the list of feature names (label-encoded, no one-hot encoding)."""
    return features

def make_predictions(faculty, students):
    features = eng_features if faculty == 'engineering' else app_features
    model = eng_model if faculty == 'engineering' else app_model
    scaler = eng_scaler if faculty == 'engineering' else app_scaler

    encoded_data = []
    for student in students:
        try:
            encoded = encode_features(student, features)
            encoded_data.append(encoded)
        except ValueError as e:
            logger.error(f"Error encoding student {student['student_id']}: {e}")
            raise

    df = pd.DataFrame(encoded_data)
    
    # Use raw feature names (label-encoded)
    expected_columns = get_expected_features(features)
    df = df.reindex(columns=expected_columns, fill_value=0)
    
    # Log DataFrame columns and values
    logger.debug(f"DataFrame columns: {list(df.columns)}")
    logger.debug(f"DataFrame shape: {df.shape}")
    if 'Exam Month' in df.columns:
        logger.debug(f"Exam Month values: {df['Exam Month'].tolist()}")
    
    # Validate numerical columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except ValueError as e:
            logger.error(f"Non-numeric values in {col}: {df[col].tolist()}")
            raise ValueError(f"Column {col} contains non-numeric values: {e}")

    # Check if scaler is valid
    try:
        if isinstance(scaler, np.ndarray):
            logger.debug(f"Scaler shape: {scaler.shape}")
            logger.debug(f"Scaler sample: {scaler[:5]}")
            # Check if scaler is invalid (e.g., contains strings or wrong shape)
            if scaler.shape == (1,) and isinstance(scaler[0], str):
                logger.warning(f"Invalid scaler for {faculty}: {scaler}. Bypassing scaling (this may affect predictions).")
                scaled_data = df.to_numpy()  # Fallback: use unscaled data
            else:
                # Validate scaler data
                if len(scaler.shape) > 1 and scaler.shape[1] != len(df.columns):
                    raise ValueError(f"Scaler columns ({scaler.shape[1]}) do not match DataFrame columns ({len(df.columns)})")
                if not np.issubdtype(scaler.dtype, np.number):
                    logger.error(f"Scaler contains non-numeric data: {scaler}")
                    raise ValueError("Scaler array contains non-numeric data")
                scaled_data = scaler[:len(df)]  # Use pre-transformed data
        else:
            scaled_data = scaler.transform(df)
    except ValueError as e:
        logger.error(f"Scaler error: {e}")
        raise ValueError(f"Scaler failed to transform data: {e}")

    # Verify scaled_data is numeric and has correct shape
    try:
        scaled_data = np.array(scaled_data, dtype=float)
        logger.debug(f"Scaled data shape: {scaled_data.shape}")
        if scaled_data.shape[1] != len(features):
            raise ValueError(f"Input shape mismatch, expected: {len(features)}, got {scaled_data.shape[1]}")
    except ValueError as e:
        logger.error(f"Scaled data error: {e}")
        raise ValueError(f"Scaled data conversion failed: {e}")

    probs = model.predict_proba(scaled_data)[:, 1]
    predictions = []
    shap_values_list = []

    # Compute SHAP values
    try:
        shap_values = eng_explainer.shap_values(scaled_data[0:1]) if faculty == 'engineering' else app_explainer.shap_values(scaled_data[0:1])
        logger.debug(f"SHAP values shape: {np.array(shap_values).shape}, type: {type(shap_values)}")
        # Handle single-output or multi-output SHAP values
        if isinstance(shap_values, list) or (isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 2):
            shap_values = shap_values[1]  # Positive class for binary classification
        # Ensure shap_values is 2D
        if len(shap_values.shape) == 3:
            shap_values = shap_values[0]
        shap_values_list.append({df.columns[i]: float(shap_values[0][i]) for i in range(len(df.columns))})
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        shap_values_list.append({col: 0.0 for col in df.columns})  # Fallback: zero SHAP values

    for i, (student, prob) in enumerate(zip(students, probs)):
        status = 'Passing' if prob > 0.55 else 'Failing' if prob < 0.45 else 'Borderline'
        intervention = 'Yes' if prob < 0.5 else 'No'
        predictions.append({
            'student_id': student['student_id'],
            'pass_probability': float(prob),
            'fail_probability': float(1 - prob),
            'status': status,
            'intervention_needed': intervention,
            'shap_values': shap_values_list[0] if i == 0 else {},
            'features': student
        })
    return predictions

def compute_metrics(predictions):
    total = len(predictions)
    at_risk = sum(1 for p in predictions if p['pass_probability'] < 0.40)
    predicted_pass = sum(1 for p in predictions if p['pass_probability'] > 0.60)
    interventions = sum(1 for p in predictions if p['intervention_needed'] == 'Yes')
    passing = sum(1 for p in predictions if p['status'] == 'Passing')
    failing = sum(1 for p in predictions if p['status'] == 'Failing')
    borderline = sum(1 for p in predictions if p['status'] == 'Borderline')
    pass_rate = (predicted_pass / total * 100) if total > 0 else 0
    risk_factors = defaultdict(int)
    for p in predictions:
        if p['pass_probability'] < 0.40:
            for feature, value in p['features'].items():
                if feature != 'student_id':
                    risk_factors[f"{feature}: {value}"] += 1
    return {
        'total_students': total,
        'at_risk': at_risk,
        'predicted_pass': predicted_pass,
        'interventions_needed': interventions,
        'faculty_performance': pass_rate,
        'risk_factors': dict(risk_factors),
        'indicators': {'Passing': passing, 'Failing': failing, 'Borderline': borderline}
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        eng_students = [generate_random_student('engineering') for _ in range(10)]
        app_students = [generate_random_student('applied_sciences') for _ in range(10)]
        logger.debug(f"Engineering students: {eng_students}")
        logger.debug(f"Applied Sciences students: {app_students}")
        eng_predictions = make_predictions('engineering', eng_students)
        app_predictions = make_predictions('applied_sciences', app_students)
        predictions_storage['engineering'].extend(eng_predictions)
        predictions_storage['applied_sciences'].extend(app_predictions)
        eng_metrics = compute_metrics(eng_predictions)
        app_metrics = compute_metrics(app_predictions)
        last_month = {
            'engineering': {k: int(v * random.uniform(0.9, 1.1)) if isinstance(v, (int, float)) else v for k, v in eng_metrics.items()},
            'applied_sciences': {k: int(v * random.uniform(0.9, 1.1)) if isinstance(v, (int, float)) else v for k, v in app_metrics.items()}
        }
        return jsonify({
            'engineering': {
                'predictions': eng_predictions,
                'metrics': eng_metrics,
                'last_month': last_month['engineering']
            },
            'applied_sciences': {
                'predictions': app_predictions,
                'metrics': app_metrics,
                'last_month': last_month['applied_sciences']
            },
            'stored_data': predictions_storage
        })
    except Exception as e:
        logger.error(f"Error in get_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stored_predictions', methods=['GET'])
def get_stored_predictions():
    return jsonify(predictions_storage)

if __name__ == '__main__':
    app.run(debug=True)