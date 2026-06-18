# Set matplotlib backend first (MUST BE FIRST IMPORT)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import traceback
from flask import Flask, render_template, request, jsonify, send_file
from openai import AzureOpenAI, OpenAI
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import shap
import pickle
import base64
from io import BytesIO
from recommender import EnhancedRecommender
from dotenv import load_dotenv
import os
import pyodbc
import seaborn as sns
import numpy as np
import re
import json
import random
import html
from datetime import datetime
from joblib import load
from collections import deque

MODELSAI_CHART_ROTATION_INDEX = 0
MODELSAI_STYLE_ROTATION_INDEX = 0
MODELSAI_RECENT_CHART_SIGNATURES = deque(maxlen=4)
MODELSAI_RECENT_PREDICTIONS = deque(maxlen=500)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Load environment variables from .env and override stale shell values.
load_dotenv(override=True)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../frontend/static', 
            template_folder='../frontend/templates')
CORS(app)

# Configure Flask
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Prevent caching issues

# Load models and scalers
engineering_model = joblib.load('Engineering (35)_optimized_model.pkl')  # Updated to new model
engineering_scaler = joblib.load('Engineering (35)_selected_scaler.pkl')  # Updated to new scaler
applied_model = joblib.load('xgboost_applied_base.pkl')
engineering_model = joblib.load('xgboost_model_cv.pkl')
applied_scaler = joblib.load('scaler_applied.pkl')
engineering_scaler = joblib.load('scaler_cv.pkl')

# Debug: Print model and scaler expected feature names
print("Engineering model expected features:", engineering_model.get_booster().feature_names)
print("Applied model expected features:", applied_model.get_booster().feature_names)
try:
    print("Engineering scaler expected features:", engineering_scaler.feature_names_in_)
    print("Applied scaler expected features:", applied_scaler.feature_names_in_)
except AttributeError:
    print("Scaler does not have feature_names_in_ attribute")

# Validate scaler features
try:
    scaler_features = engineering_scaler.feature_names_in_
    if set(scaler_features) != set(engineering_model.get_booster().feature_names):
        print("Warning: Engineering scaler features do not match model features!")
        print("Scaler features:", scaler_features)
        print("Model features:", engineering_model.get_booster().feature_names)
except AttributeError:
    print("Cannot validate engineering scaler features: feature_names_in_ not available")

try:
    scaler_features = applied_scaler.feature_names_in_
    if set(scaler_features) != set(applied_model.get_booster().feature_names):
        print("Warning: Applied scaler features do not match model features!")
        print("Scaler features:", scaler_features)
        print("Model features:", applied_model.get_booster().feature_names)
except AttributeError:
    print("Cannot validate applied scaler features: feature_names_in_ not available")

# Load dataset for recommendations
with open('combined_student_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Initialize recommender
recommender = EnhancedRecommender()
recommender.load_data(data)

# Define categorical columns with feature names aligned to model
engineering_features = [
    'Residence', 'FTEN Status', 'Bursary', 'Offering Type', 'Disability',
    'Age Group', 'Exam Type', 'Exam Month', 'Race', 'Gender', 'Stud_Type',
    'Marital Status'
]

applied_features = [
    'Exam Type', 'Gender', 'Race', 'Disability', 'Department', 'Final Year',
    'FTEN Status', 'Residence', 'Marital Status', 'Exam Month', 'Age Group',
    'Stud Type', 'Bursary', 'Offer Type'
]

engineering_categories = {
    'Residence': ['Y', 'N'],
    'FTEN Status': ['N', 'F', 'T', 'E'],
    'Bursary': ['N', 'Y'],
    'Offering Type': ['D1', 'D3', 'P1', 'I1', 'P3', 'I3', 'D2', 'D6', 'D8', 'I7', 'D7', 'P7'],
    'Disability': ['N', 'Y'],
    'Age Group': ['15-19', '20-29', '30-39', '40-49', '>50'],
    'Exam Type': ['NORMAL EXAM', 'RE-EXAM(SUPP-MIDYEAR/YEAR END)', 'SPECIAL EXAM', 'EXPERIENTIAL TRAINING'],
    'Race': ['AFRICAN', 'INDIAN', 'WHITE', 'COLOURED', 'NON SA'],
    'Gender': ['F', 'M'],
    'Stud_Type': ['NORMAL STUDENT', 'SADC STUDENT', 'INTERNATIONAL STUDENT', 'REFUGEE', 'ASYLUM SEEKER'],
    'Marital Status': ['M', 'S']
}

applied_categories = {
    'Gender': ['M', 'F'],
    'Department': ['MARITIME STUDIES', 'BIOTECHNOLOGY & FOOD SCIENCE', 'HORTICULTURE', 'CHEMISTRY',
                   'DEPT OF TEXTILE SC & APP TECH', 'SPORT STUDIES', 'FOOD & NUTRIT CONSUMER SCIENCE',
                   'MATHEMATICS', 'FACULTY OFFICE-APPLIED SCIENCE'],
    'Residence': ['Y', 'N'],
    'Race': ['AFRICAN', 'INDIAN', 'WHITE', 'COLOURED', 'NON SA'],
    'Disability': ['N', 'Y'],
    'Exam Type': ['NORMAL EXAM', 'RE-EXAM(SUPP-MIDYEAR/YEAR END)', 'SPECIAL EXAM', 'EXPERIENTIAL TRAINING'],
    'Stud Type': ['NORMAL STUDENT', 'SADC STUDENT', 'INTERNATIONAL STUDENT', 'REFUGEE', 'ASYLUM SEEKER'],
    'Age Group': ['15-19', '20-29', '30-39', '40-49', '>50'],
    'Bursary': ['N', 'Y'],
    'Offer Type': ['D1', 'D3', 'P1', 'I1', 'P3', 'I3', 'D2', 'D6', 'D8', 'I7', 'D7', 'P7'],
    'Final Year': ['Y', 'N'],
    'Marital Status': ['MARRIED', 'SINGLE', 'SI'],
    'FTEN Status': ['N', 'F', 'T', 'E']
}

feature_aliases = {
    'Residence': ['Residence', 'Residence Flag', 'ResidenceIndicator'],
    'FTEN Status': ['FTEN Status', 'FTEN_Status', 'StudentType', 'FullPartTime'],
    'Bursary': ['Bursary', 'Bursary Flag', 'BursaryFlag'],
    'Offering Type': ['Offering Type', 'Offer Type', 'OfferingType'],
    'Disability': ['Disability', 'Disability Flag', 'DisabilityFlag'],
    'Age Group': ['Age Group', 'AgeGroup'],
    'Exam Type': ['Exam Type', 'ExamType'],
    'Exam Month': ['Exam Month', 'ExamMonth', 'exam_month'],
    'Race': ['Race', 'Ethnic Group Name', 'EthnicGroupName'],
    'Gender': ['Gender'],
    'Stud_Type': ['Stud_Type', 'Stud Type', 'StudentTypeDescription'],
    'Marital Status': ['Marital Status', 'MaritalStatus'],
    'Department': ['Department'],
    'Final Year': ['Final Year', 'FinalYear'],
    'Stud Type': ['Stud Type', 'Stud_Type', 'StudentTypeDescription'],
    'Offer Type': ['Offer Type', 'Offering Type', 'OfferingType']
}


def _to_dataframe(records):
    """Coerce loaded pickle/csv data into a DataFrame for sampling."""
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, list):
        if len(records) > 0 and isinstance(records[0], dict):
            return pd.DataFrame(records)
        return pd.DataFrame({'value': records})
    if isinstance(records, dict):
        try:
            return pd.DataFrame(records)
        except Exception:
            if 'data' in records and isinstance(records['data'], list):
                return pd.DataFrame(records['data'])
    return pd.DataFrame()


realtime_source_df = _to_dataframe(data)


def _get_row_value(row, feature_name):
    """Fetch a value from a sampled row using feature aliases."""
    aliases = feature_aliases.get(feature_name, [feature_name])
    for alias in aliases:
        if alias in row and pd.notna(row[alias]):
            return row[alias]
    return None


def _normalize_exam_month(raw_value):
    try:
        month = int(float(raw_value))
        return month if 1 <= month <= 12 else random.randint(1, 12)
    except Exception:
        return random.randint(1, 12)


def _match_allowed_value(raw_value, allowed_values):
    if raw_value is None or pd.isna(raw_value):
        return random.choice(allowed_values)

    value = str(raw_value).strip()
    upper = value.upper()

    synonym_map = {
        'YES': 'Y',
        'NO': 'N',
        'TRUE': 'Y',
        'FALSE': 'N',
        'MALE': 'M',
        'FEMALE': 'F',
        'MARRIED': 'MARRIED',
        'SINGLE': 'SINGLE',
        'SI': 'SI',
        'NORMAL': 'NORMAL EXAM',
        'RE-EXAM': 'RE-EXAM(SUPP-MIDYEAR/YEAR END)',
        'SPECIAL': 'SPECIAL EXAM'
    }
    if upper in synonym_map:
        candidate = synonym_map[upper]
        for allowed in allowed_values:
            if str(allowed).upper() == candidate:
                return allowed

    for allowed in allowed_values:
        if value == str(allowed):
            return allowed
        if upper == str(allowed).upper():
            return allowed

    return random.choice(allowed_values)


def _sample_students_from_real_data(faculty, sample_size=10):
    """Randomly sample real rows and map them to model feature dictionaries."""
    features = engineering_features if faculty == 'engineering' else applied_features
    categories = engineering_categories if faculty == 'engineering' else applied_categories

    if realtime_source_df.empty:
        sampled_rows = [{} for _ in range(sample_size)]
    else:
        sampled = realtime_source_df.sample(n=sample_size, replace=len(realtime_source_df) < sample_size)
        sampled_rows = sampled.to_dict('records')

    students = []
    for row in sampled_rows:
        student = {}
        for feature in features:
            raw_value = _get_row_value(row, feature)
            if feature == 'Exam Month':
                student[feature] = _normalize_exam_month(raw_value)
            else:
                allowed = categories.get(feature, [])
                student[feature] = _match_allowed_value(raw_value, allowed) if allowed else str(raw_value)

        student_id = _get_row_value(row, 'Student Number')
        if student_id is None:
            student_id = _get_row_value(row, 'student_id')
        if student_id is None:
            student_id = random.randint(10000000, 99999999)

        students.append({
            'student_id': str(student_id),
            'features': student
        })

    return students


def _predict_faculty_batch(faculty, sampled_students):
    """Run batch predictions using the frozen model + scaler for a faculty."""
    features = engineering_features if faculty == 'engineering' else applied_features
    categories = engineering_categories if faculty == 'engineering' else applied_categories
    model = engineering_model if faculty == 'engineering' else applied_model
    scaler = engineering_scaler if faculty == 'engineering' else applied_scaler

    raw_df = pd.DataFrame([entry['features'] for entry in sampled_students])

    encoded_df = raw_df.copy()
    for col, allowed_values in categories.items():
        if col in encoded_df.columns:
            le = LabelEncoder()
            le.fit(allowed_values)
            safe_values = encoded_df[col].apply(lambda x: _match_allowed_value(x, allowed_values))
            encoded_df[col] = le.transform(safe_values.astype(str))

    try:
        scaler_feature_order = list(scaler.feature_names_in_)
    except AttributeError:
        scaler_feature_order = features

    for col in scaler_feature_order:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
    encoded_df = encoded_df[scaler_feature_order]

    scaled = scaler.transform(encoded_df)
    model_df = pd.DataFrame(scaled, columns=scaler_feature_order)

    expected_features = model.get_booster().feature_names
    for col in expected_features:
        if col not in model_df.columns:
            model_df[col] = 0
    model_df = model_df[expected_features]

    probs = model.predict_proba(model_df)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(model_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    output = []
    for idx, row in enumerate(sampled_students):
        pass_probability = float(probs[idx][1])
        fail_probability = float(probs[idx][0])
        result = 'Passing' if pass_probability > 0.55 else 'Failing' if pass_probability < 0.45 else 'Borderline'

        output.append({
            'student_id': row['student_id'],
            'status': result,
            'result': result,
            'pass_probability': pass_probability,
            'fail_probability': fail_probability,
            'confidence': float(abs(pass_probability - fail_probability)),
            'intervention_needed': 'Yes' if pass_probability < 0.5 else 'No',
            'shap_values': {feature: float(value) for feature, value in zip(expected_features, shap_values[idx])},
            'features': row['features'],
            'student_data': row['features'],
            'faculty': faculty,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })

    return output


def _register_modelsai_prediction_event(event):
    """Store recent prediction events so ModelsAI can ground on newest predictions immediately."""
    if not isinstance(event, dict):
        return

    normalized = {
        'student_id': str(event.get('student_id', 'unknown')),
        'faculty': event.get('faculty', 'unknown'),
        'result': event.get('result', event.get('status', 'Unknown')),
        'pass_probability': float(event.get('pass_probability', 0.0) or 0.0),
        'fail_probability': float(event.get('fail_probability', 0.0) or 0.0),
        'confidence': float(event.get('confidence', 0.0) or 0.0),
        'intervention_needed': event.get('intervention_needed', 'No'),
        'shap_values': event.get('shap_values', {}) if isinstance(event.get('shap_values'), dict) else {},
        'features': event.get('features', {}) if isinstance(event.get('features'), dict) else {},
        'timestamp': event.get('timestamp') or (datetime.utcnow().isoformat() + 'Z')
    }
    MODELSAI_RECENT_PREDICTIONS.append(normalized)


def _merge_recent_predictions_into_context(context):
    """Blend latest prediction events into grounding context to keep summaries current."""
    if not MODELSAI_RECENT_PREDICTIONS:
        return context

    recent_df = pd.DataFrame(list(MODELSAI_RECENT_PREDICTIONS))
    if recent_df.empty:
        return context

    context = context if isinstance(context, dict) else {}
    summary = context.setdefault('summary', {})

    prediction_counts = recent_df['result'].value_counts(dropna=False).to_dict() if 'result' in recent_df.columns else {}
    faculty_counts = recent_df['faculty'].value_counts(dropna=False).to_dict() if 'faculty' in recent_df.columns else {}

    mix = {}
    if 'faculty' in recent_df.columns and 'result' in recent_df.columns:
        mix = pd.crosstab(recent_df['faculty'], recent_df['result'], dropna=False).to_dict(orient='index')

    shap_scores = {}
    if 'shap_values' in recent_df.columns:
        for row_shap in recent_df['shap_values']:
            if not isinstance(row_shap, dict):
                continue
            for feature, value in row_shap.items():
                shap_scores[feature] = shap_scores.get(feature, 0.0) + abs(float(value))

    shap_top_features = [
        {'feature': feature, 'importance': float(score)}
        for feature, score in sorted(shap_scores.items(), key=lambda kv: kv[1], reverse=True)[:12]
    ]

    top_risk_students = []
    if 'fail_probability' in recent_df.columns:
        fail_probs = pd.to_numeric(recent_df['fail_probability'], errors='coerce')
        risk_df = recent_df.loc[fail_probs.notna()].copy()
        risk_df['fail_probability'] = pd.to_numeric(risk_df['fail_probability'], errors='coerce')
        risk_df = risk_df.sort_values('fail_probability', ascending=False).head(12)
        risk_cols = [c for c in ['student_id', 'faculty', 'result', 'fail_probability', 'pass_probability', 'confidence', 'intervention_needed', 'timestamp'] if c in risk_df.columns]
        top_risk_students = risk_df[risk_cols].fillna('').to_dict('records')

    summary['recent_prediction_count'] = int(len(recent_df))
    summary['recent_prediction_counts'] = prediction_counts
    summary['recent_faculty_counts'] = faculty_counts
    summary['recent_prediction_by_faculty'] = mix
    summary['recent_top_risk_students'] = top_risk_students
    if shap_top_features:
        summary['recent_shap_top_features'] = shap_top_features

    base_prediction_counts = summary.get('prediction_counts') or {}
    merged_prediction_counts = dict(base_prediction_counts)
    for k, v in prediction_counts.items():
        merged_prediction_counts[k] = int(merged_prediction_counts.get(k, 0)) + int(v)
    summary['prediction_counts'] = merged_prediction_counts

    base_faculty_counts = summary.get('faculty_counts') or {}
    merged_faculty_counts = dict(base_faculty_counts)
    for k, v in faculty_counts.items():
        merged_faculty_counts[k] = int(merged_faculty_counts.get(k, 0)) + int(v)
    summary['faculty_counts'] = merged_faculty_counts

    base_mix = summary.get('prediction_by_faculty') or {}
    merged_mix = {k: dict(v) if isinstance(v, dict) else {} for k, v in base_mix.items()}
    for faculty, result_map in mix.items():
        merged_mix.setdefault(faculty, {})
        for result, count in (result_map or {}).items():
            merged_mix[faculty][result] = int(merged_mix[faculty].get(result, 0)) + int(count)
    summary['prediction_by_faculty'] = merged_mix

    existing_top_risk = summary.get('top_risk_students') or []
    summary['top_risk_students'] = (top_risk_students + existing_top_risk)[:12]

    if shap_top_features:
        existing_shap = summary.get('shap_top_features') or []
        shap_map = {item.get('feature'): float(item.get('importance', 0.0)) for item in existing_shap if isinstance(item, dict) and item.get('feature')}
        for item in shap_top_features:
            feature = item.get('feature')
            if not feature:
                continue
            shap_map[feature] = shap_map.get(feature, 0.0) + float(item.get('importance', 0.0))
        summary['shap_top_features'] = [
            {'feature': feature, 'importance': float(score)}
            for feature, score in sorted(shap_map.items(), key=lambda kv: kv[1], reverse=True)[:12]
        ]

    base_rows = context.get('sample_rows') or []
    recent_rows = recent_df.head(12).fillna('').to_dict('records')
    context['sample_rows'] = (recent_rows + base_rows)[:20]
    context['record_count'] = int((context.get('record_count') or 0) + len(recent_rows))

    options = context.get('visualization_options') or []
    options.extend([
        {'type': 'bar', 'title': 'Recent prediction distribution', 'x': 'result', 'y': 'Count'},
        {'type': 'bar', 'title': 'Recent predictions by faculty', 'x': 'faculty', 'y': 'Count'}
    ])
    context['visualization_options'] = options

    context['source'] = f"{context.get('source', 'context')}+recent_events"
    return context


def _parse_sample_size(value, default_size=10, min_size=1, max_size=100):
    """Parse and clamp a requested sample size to safe bounds."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default_size
    return max(min_size, min(parsed, max_size))


@app.route('/api/realtime-predictions', methods=['GET'])
def realtime_predictions():
    """Generate real-time predictions by sampling real data and running frozen models."""
    try:
        default_size = _parse_sample_size(request.args.get('size'), default_size=10)
        engineering_size = _parse_sample_size(request.args.get('engineering_size'), default_size=default_size)
        applied_size = _parse_sample_size(request.args.get('applied_size'), default_size=default_size)

        engineering_sample = _sample_students_from_real_data('engineering', sample_size=engineering_size)
        applied_sample = _sample_students_from_real_data('applied_sciences', sample_size=applied_size)

        engineering_predictions = _predict_faculty_batch('engineering', engineering_sample)
        applied_predictions = _predict_faculty_batch('applied_sciences', applied_sample)
        for item in engineering_predictions + applied_predictions:
            _register_modelsai_prediction_event(item)

        return jsonify({
            'engineering': engineering_predictions,
            'applied_sciences': applied_predictions,
            'meta': {
                'source_rows': int(len(realtime_source_df)),
                'requested_size': default_size,
                'engineering_size': engineering_size,
                'applied_size': applied_size,
                'generated_at': datetime.utcnow().isoformat() + 'Z'
            }
        }), 200
    except Exception as e:
        print(f"Realtime predictions error: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Database connection
def get_db_connection():
    connection_string = os.getenv('CONNECTION_STRING')
    try:
        timeout = int(os.getenv('DB_CONNECTION_TIMEOUT', '3'))
        return pyodbc.connect(connection_string, timeout=timeout)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# Helper functions
def create_shap_plot(shap_values, feature_names):
    """Create SHAP plot and return as base64 image"""
    try:
        plt.figure(figsize=(10, 6))
        sorted_idx = shap_values.argsort()
        shap_values_sorted = shap_values[sorted_idx]
        feature_names_sorted = [feature_names[i] for i in sorted_idx]
        colors = ['#0d16c8' if x >= 0 else '#ff0000' for x in shap_values_sorted]
        plt.barh(feature_names_sorted, shap_values_sorted, color=colors)
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title('Feature Contributions')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"Error creating SHAP plot: {e}")
        return None
    finally:
        plt.close('all')

def create_data_plot(df, x_col, y_col=None, plot_type='bar'):
    """Create various plots from dataframe"""
    try:
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'bar':
            sns.barplot(x=x_col, y=y_col, data=df) if y_col else df[x_col].value_counts().plot(kind='bar')
        elif plot_type == 'line':
            sns.lineplot(x=x_col, y=y_col, data=df) if y_col else None
        elif plot_type == 'hist':
            sns.histplot(df[x_col], kde=True)
        elif plot_type == 'pie':
            df[x_col].value_counts().plot(kind='pie', autopct='%1.1f%%')
        
        plt.title(f'{plot_type.capitalize()} Plot of {x_col}' + (f' vs {y_col}' if y_col else ''))
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        print(f"Error creating {plot_type} plot: {e}")
        return None
    finally:
        plt.close('all')

def execute_query(query):
    """Execute SQL query and return DataFrame"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            df = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
            conn.close()
            return df
        return None
    except Exception as e:
        print(f"Query execution error: {e}")
        return None

def get_database_schema():
    """Retrieve complete database schema"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    SCHEMA_NAME(o.schema_id) AS schema_name,
                    o.name AS object_name,
                    o.type_desc AS object_type,
                    c.name AS column_name,
                    t.name AS data_type,
                    c.is_nullable
                FROM sys.objects o
                JOIN sys.columns c ON o.object_id = c.object_id
                JOIN sys.types t ON c.user_type_id = t.user_type_id
                WHERE o.type IN ('U', 'V')
                ORDER BY schema_name, object_name, c.column_id
            """)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            schema = {}
            for row in rows:
                full_name = f"{row[0]}.{row[1]}"
                if full_name not in schema:
                    schema[full_name] = {'type': row[2], 'columns': []}
                schema[full_name]['columns'].append({
                    'name': row[3],
                    'type': row[4],
                    'nullable': bool(row[5])
                })
            return schema
        return None
    except Exception as e:
        print(f"Schema query error: {e}")
        return None

def generate_sql(nl_query, schema):
    """Generate SQL from natural language using AI"""
    try:
        client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('OPENAI_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
        )
        
        schema_info = "Database schema:\n"
        for obj_name, obj_info in schema.items():
            schema_info += f"- {obj_name} ({obj_info['type']}):\n"
            for col in obj_info['columns']:
                schema_info += f"  * {col['name']} ({col['type']})\n"
        
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOY_NAME'),
            messages=[
                {"role": "system", "content": f"""
                You are a SQL expert for SQL Server. Convert natural language to SQL.
                E.g 
                1. Who are the students predicted to fail with high confidence?
                   Answer: SELECT [Student Number], [First Names], [Prediction], [Confidence_Label], [Probability_Fail]
                            FROM [ITS_Views_Warehouse].[dbo].[STUD_ML_PREDICTIONS]
                            WHERE [Prediction] = 'Fail' AND [Confidence_Label] = 'High'
                            ORDER BY [Probability_Fail] DESC;
                2. Are students in residence more likely to pass according to predictions?
                   Answer: SELECT [Residence], 
                          AVG(CAST([Prediction] = 'Pass' AS FLOAT)) AS Pass_Rate
                          FROM [ITS_Views_Warehouse].[dbo].[STUD_ML_PREDICTIONS]
                          GROUP BY [Residence];

                Only use these database objects:
                {schema_info}
                Rules:
                1. Use exact table/view names
                2. Include schema name (e.g., dbo.table)
                3. Only use existing columns
                """},
                {"role": "user", "content": nl_query}
            ],
            temperature=0.3
        )
        
        sql = response.choices[0].message.content
        sql = re.sub(r'```sql|```', '', sql).strip()
        
        if not any(obj.lower() in sql.lower() for obj in schema.keys()):
            raise ValueError(f"Query must use one of: {', '.join(schema.keys())}")
        
        return sql
    except Exception as e:
        print(f"AI SQL generation error: {e}")
        raise

def analyze_data(df):
    """Generate insights from DataFrame"""
    if df.empty:
        return "No data available."
    
    insights = [f"Dataset: {len(df)} records, {len(df.columns)} columns."]
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            insights.append(
                f"{col}: Mean={df[col].mean():.2f}, "
                f"Median={df[col].median():.2f}, "
                f"Range=[{df[col].min():.2f}-{df[col].max():.2f}]"
            )
        else:
            top = df[col].value_counts().head(3)
            insights.append(f"{col}: Top values - {', '.join([f'{k} ({v})' for k, v in top.items()])}")
    
    return "\n".join(insights)

def format_ai_response(response_text):
    """Format AI response with HTML"""
    paragraphs = response_text.split('\n\n')
    html = "<div style='font-family: Arial, sans-serif; line-height: 1.6; color: #333;'>"
    
    for p in paragraphs:
        if p.startswith("####"):
            html += f"<h6 style='color: #119a61; margin-top: 20px;'>{p.replace('####', '').strip()}</h6>"
        elif p.startswith("-"):
            html += "<ul>" + "".join(f"<li>{item.replace('-', '').strip()}</li>" 
                    for item in p.split('\n') if item.strip()) + "</ul>"
        else:
            html += f"<p>{p.strip()}</p>"
    
    return html + "</div>"


def get_modelsai_settings():
    """Centralized ModelsAI runtime settings sourced from environment variables."""
    return {
        'provider': 'openai',
        'temperature': float(os.getenv('MODELSAI_TEMPERATURE', '0.35')),
        'max_tokens': int(os.getenv('MODELSAI_MAX_TOKENS', '1400')),
        'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
        'openai_base_url': os.getenv('OPENAI_BASE_URL', '').strip()
    }


def create_modelsai_client(provider):
    """Create vanilla OpenAI client (ModelsAI is OpenAI-only)."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY is missing')

    base_url = os.getenv('OPENAI_BASE_URL', '').strip()
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    return client, model_name


def _is_small_talk_message(message):
    text = (message or '').strip().lower()
    if not text:
        return False

    normalized = re.sub(r'[^a-z0-9\s]', '', text)
    small_talk_phrases = {
        'hi', 'hello', 'hey', 'yo', 'how are you', 'hows it going', 'good morning',
        'good afternoon', 'good evening', 'whats up', 'how do you do'
    }

    if normalized in small_talk_phrases:
        return True

    tokens = normalized.split()
    return len(tokens) <= 3 and any(token in {'hi', 'hello', 'hey'} for token in tokens)


def _small_talk_reply():
    return (
        "<div style='font-family: Segoe UI, Arial, sans-serif; line-height: 1.6; color: #1f2933;'>"
        "<p style='margin: 8px 0;'>I am good, thanks for asking. "
        "I can help with student-risk insights, predictions, and dashboard summaries whenever you are ready.</p>"
        "</div>"
    )


def _safe_column(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match:
            return match
    return None


def _dashboard_context_from_db(limit=2000):
    """Build grounding context from prediction warehouse view when available."""
    schema = get_database_schema() or {}
    view_name = None
    for obj_name in schema.keys():
        if 'stud_ml_predictions' in obj_name.lower():
            view_name = obj_name
            break

    if not view_name:
        return None

    columns = [c['name'] for c in schema[view_name]['columns']]
    pred_col = _safe_column(columns, ['Prediction'])
    faculty_col = _safe_column(columns, ['Faculty', 'Faculty_Name', 'Faculty Name'])
    conf_col = _safe_column(columns, ['Confidence_Label', 'Confidence Label'])
    fail_prob_col = _safe_column(columns, ['Probability_Fail', 'Fail_Probability', 'Probability Fail'])
    res_col = _safe_column(columns, ['Residence', 'Residence Flag', 'ResidenceIndicator'])
    student_col = _safe_column(columns, ['Student Number', 'Student_Number', 'student_id'])
    name_col = _safe_column(columns, ['First Names', 'First_Name', 'Name'])

    select_cols = [c for c in [student_col, name_col, pred_col, faculty_col, conf_col, fail_prob_col, res_col] if c]
    if not select_cols:
        return None

    select_sql = ', '.join([f'[{c}]' for c in select_cols])
    schema_name, object_name = view_name.split('.', 1)
    sql = f"SELECT TOP {limit} {select_sql} FROM [{schema_name}].[{object_name}]"
    df = execute_query(sql)
    if df is None or df.empty:
        return None

    context = {
        'source': view_name,
        'record_count': int(len(df)),
        'columns_used': select_cols,
        'summary': {},
        'sample_rows': df.head(20).fillna('').to_dict('records'),
        'visualization_options': []
    }

    if pred_col and pred_col in df.columns:
        context['summary']['prediction_counts'] = df[pred_col].value_counts(dropna=False).to_dict()
        context['visualization_options'].append({
            'type': 'bar',
            'title': 'Prediction distribution',
            'x': pred_col,
            'y': 'Count'
        })

    if faculty_col and faculty_col in df.columns:
        context['summary']['faculty_counts'] = df[faculty_col].value_counts(dropna=False).to_dict()
        context['visualization_options'].append({
            'type': 'bar',
            'title': 'Predictions by faculty',
            'x': faculty_col,
            'y': 'Count'
        })

    if conf_col and conf_col in df.columns:
        context['summary']['confidence_counts'] = df[conf_col].value_counts(dropna=False).to_dict()

    if fail_prob_col and fail_prob_col in df.columns:
        numeric_fail = pd.to_numeric(df[fail_prob_col], errors='coerce')
        context['summary']['avg_fail_probability'] = float(numeric_fail.mean()) if numeric_fail.notna().any() else None
        context['summary']['fail_probability_values'] = [float(v) for v in numeric_fail.dropna().head(400).tolist()]
        high_risk = df.loc[numeric_fail.notna()].copy()
        high_risk['__fail_prob__'] = pd.to_numeric(high_risk[fail_prob_col], errors='coerce')
        high_risk = high_risk.sort_values('__fail_prob__', ascending=False).head(12)
        risk_cols = [c for c in [student_col, name_col, faculty_col, pred_col, fail_prob_col] if c and c in high_risk.columns]
        context['summary']['top_risk_students'] = high_risk[risk_cols].fillna('').to_dict('records')
        context['visualization_options'].append({
            'type': 'histogram',
            'title': 'Fail probability distribution',
            'x': fail_prob_col,
            'y': 'Frequency'
        })

    if pred_col and faculty_col and pred_col in df.columns and faculty_col in df.columns:
        pivot = pd.crosstab(df[faculty_col], df[pred_col], dropna=False)
        context['summary']['prediction_by_faculty'] = pivot.to_dict(orient='index')
        context['visualization_options'].append({
            'type': 'stacked_bar',
            'title': 'Prediction mix by faculty',
            'x': faculty_col,
            'series': pred_col
        })

    if faculty_col and fail_prob_col and faculty_col in df.columns and fail_prob_col in df.columns:
        risk_df = df[[faculty_col, fail_prob_col]].copy()
        risk_df[fail_prob_col] = pd.to_numeric(risk_df[fail_prob_col], errors='coerce')
        risk_df = risk_df.dropna(subset=[fail_prob_col])
        if not risk_df.empty:
            faculty_avg = risk_df.groupby(faculty_col)[fail_prob_col].mean().to_dict()
            context['summary']['faculty_avg_fail_probability'] = {str(k): float(v) for k, v in faculty_avg.items()}

    return context


def _dashboard_context_fallback():
    """Fallback grounding context using live model-generated prediction snapshots."""
    try:
        engineering_sample = _sample_students_from_real_data('engineering', sample_size=60)
        applied_sample = _sample_students_from_real_data('applied_sciences', sample_size=60)
        engineering_preds = _predict_faculty_batch('engineering', engineering_sample)
        applied_preds = _predict_faculty_batch('applied_sciences', applied_sample)
        all_preds = engineering_preds + applied_preds

        if not all_preds:
            raise ValueError('No predictions generated for fallback context')

        pred_df = pd.DataFrame(all_preds)
        prediction_counts = pred_df['result'].value_counts(dropna=False).to_dict() if 'result' in pred_df.columns else {}
        faculty_counts = pred_df['faculty'].value_counts(dropna=False).to_dict() if 'faculty' in pred_df.columns else {}
        risk_threshold = float(os.getenv('MODELSAI_HIGH_RISK_THRESHOLD', '0.65'))

        mix = {}
        if 'faculty' in pred_df.columns and 'result' in pred_df.columns:
            mix = pd.crosstab(pred_df['faculty'], pred_df['result'], dropna=False).to_dict(orient='index')

        shap_scores = {}
        if 'shap_values' in pred_df.columns:
            for row_shap in pred_df['shap_values']:
                if not isinstance(row_shap, dict):
                    continue
                for feature, value in row_shap.items():
                    shap_scores[feature] = shap_scores.get(feature, 0.0) + abs(float(value))

        shap_top_features = [
            {'feature': feature, 'importance': float(score)}
            for feature, score in sorted(shap_scores.items(), key=lambda kv: kv[1], reverse=True)[:12]
        ]

        top_risk_students = []
        if 'fail_probability' in pred_df.columns:
            fail_probs = pd.to_numeric(pred_df['fail_probability'], errors='coerce')
            risk_df = pred_df.loc[fail_probs.notna()].copy()
            risk_df['fail_probability'] = pd.to_numeric(risk_df['fail_probability'], errors='coerce')
            risk_df = risk_df.sort_values('fail_probability', ascending=False).head(12)
            risk_cols = [c for c in ['student_id', 'faculty', 'result', 'fail_probability', 'pass_probability', 'confidence', 'intervention_needed'] if c in risk_df.columns]
            top_risk_students = risk_df[risk_cols].fillna('').to_dict('records')

        high_risk_count = 0
        if 'fail_probability' in pred_df.columns:
            fail_probs = pd.to_numeric(pred_df['fail_probability'], errors='coerce')
            high_risk_count = int((fail_probs >= risk_threshold).sum())

        faculty_avg_fail_probability = {}
        if 'faculty' in pred_df.columns and 'fail_probability' in pred_df.columns:
            risk_df = pred_df[['faculty', 'fail_probability']].copy()
            risk_df['fail_probability'] = pd.to_numeric(risk_df['fail_probability'], errors='coerce')
            risk_df = risk_df.dropna(subset=['fail_probability'])
            if not risk_df.empty:
                faculty_avg_fail_probability = {
                    str(k): float(v)
                    for k, v in risk_df.groupby('faculty')['fail_probability'].mean().to_dict().items()
                }

        return {
            'source': 'live_model_predictions_fallback',
            'record_count': int(len(pred_df)),
            'columns_used': list(pred_df.columns),
            'summary': {
                'note': 'Using live model-generated predictions as fallback context.',
                'prediction_counts': prediction_counts,
                'faculty_counts': faculty_counts,
                'prediction_by_faculty': mix,
                'avg_fail_probability': float(pd.to_numeric(pred_df.get('fail_probability', pd.Series(dtype=float)), errors='coerce').mean()) if 'fail_probability' in pred_df.columns else None,
                'high_risk_threshold': risk_threshold,
                'high_risk_count': high_risk_count,
                'top_risk_students': top_risk_students,
                'shap_top_features': shap_top_features,
                'fail_probability_values': [float(v) for v in pd.to_numeric(pred_df.get('fail_probability', pd.Series(dtype=float)), errors='coerce').dropna().head(400).tolist()] if 'fail_probability' in pred_df.columns else [],
                'confidence_values': [float(v) for v in pd.to_numeric(pred_df.get('confidence', pd.Series(dtype=float)), errors='coerce').dropna().head(400).tolist()] if 'confidence' in pred_df.columns else [],
                'faculty_avg_fail_probability': faculty_avg_fail_probability
            },
            'sample_rows': pred_df.head(20).fillna('').to_dict('records'),
            'visualization_options': [
                {'type': 'bar', 'title': 'Prediction distribution', 'x': 'result', 'y': 'Count'},
                {'type': 'pie', 'title': 'Passing vs Borderline vs Failing', 'x': 'result', 'y': 'Count'},
                {'type': 'bar', 'title': 'Predictions by faculty', 'x': 'faculty', 'y': 'Count'},
                {'type': 'stacked_bar', 'title': 'Prediction mix by faculty', 'x': 'faculty', 'series': 'result'},
                {'type': 'barh', 'title': 'Top SHAP feature impact', 'x': 'importance', 'y': 'feature'}
            ]
        }
    except Exception as e:
        if realtime_source_df is None or realtime_source_df.empty:
            return {
                'source': 'realtime_source_df',
                'record_count': 0,
                'columns_used': [],
                'summary': {'note': f'No dashboard source data available. Fallback failed: {e}'},
                'sample_rows': [],
                'visualization_options': []
            }

        sample_df = realtime_source_df.head(20).copy()
        return {
            'source': 'realtime_source_df',
            'record_count': int(len(realtime_source_df)),
            'columns_used': list(realtime_source_df.columns),
            'summary': {
                'note': f'Using local source dataset fallback. Live prediction fallback failed: {e}',
                'numeric_columns': [c for c in realtime_source_df.columns if pd.api.types.is_numeric_dtype(realtime_source_df[c])][:15]
            },
            'sample_rows': sample_df.fillna('').to_dict('records'),
            'visualization_options': []
        }


def _merge_count_dicts(left, right):
    merged = dict(left or {})
    for key, value in (right or {}).items():
        merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def _merge_nested_count_dicts(left, right):
    merged = {k: dict(v) if isinstance(v, dict) else {} for k, v in (left or {}).items()}
    for outer_key, inner_map in (right or {}).items():
        merged.setdefault(outer_key, {})
        for inner_key, count in (inner_map or {}).items():
            merged[outer_key][inner_key] = int(merged[outer_key].get(inner_key, 0)) + int(count)
    return merged


def _combine_modelsai_contexts(primary, secondary):
    if not primary:
        return secondary
    if not secondary:
        return primary

    primary_summary = primary.get('summary') or {}
    secondary_summary = secondary.get('summary') or {}

    primary_summary['prediction_counts'] = _merge_count_dicts(
        primary_summary.get('prediction_counts'),
        secondary_summary.get('prediction_counts')
    )
    primary_summary['faculty_counts'] = _merge_count_dicts(
        primary_summary.get('faculty_counts'),
        secondary_summary.get('faculty_counts')
    )
    primary_summary['prediction_by_faculty'] = _merge_nested_count_dicts(
        primary_summary.get('prediction_by_faculty'),
        secondary_summary.get('prediction_by_faculty')
    )

    primary_top_risk = primary_summary.get('top_risk_students') or []
    secondary_top_risk = secondary_summary.get('top_risk_students') or []
    primary_summary['top_risk_students'] = (primary_top_risk + secondary_top_risk)[:12]

    if secondary_summary.get('shap_top_features'):
        shape_map = {
            item.get('feature'): float(item.get('importance', 0.0))
            for item in (primary_summary.get('shap_top_features') or [])
            if isinstance(item, dict) and item.get('feature')
        }
        for item in secondary_summary.get('shap_top_features') or []:
            feature = item.get('feature')
            if not feature:
                continue
            shape_map[feature] = shape_map.get(feature, 0.0) + float(item.get('importance', 0.0))
        primary_summary['shap_top_features'] = [
            {'feature': feature, 'importance': float(score)}
            for feature, score in sorted(shape_map.items(), key=lambda kv: kv[1], reverse=True)[:12]
        ]

    primary_summary['fail_probability_values'] = (
        (primary_summary.get('fail_probability_values') or []) +
        (secondary_summary.get('fail_probability_values') or [])
    )[:400]

    primary_summary['confidence_values'] = (
        (primary_summary.get('confidence_values') or []) +
        (secondary_summary.get('confidence_values') or [])
    )[:400]

    primary_summary['faculty_avg_fail_probability'] = {
        **(primary_summary.get('faculty_avg_fail_probability') or {}),
        **(secondary_summary.get('faculty_avg_fail_probability') or {})
    }

    primary['summary'] = primary_summary
    primary['sample_rows'] = ((primary.get('sample_rows') or []) + (secondary.get('sample_rows') or []))[:20]
    primary['columns_used'] = list(dict.fromkeys((primary.get('columns_used') or []) + (secondary.get('columns_used') or [])))
    primary['record_count'] = int((primary.get('record_count') or 0) + (secondary.get('record_count') or 0))
    primary['visualization_options'] = (primary.get('visualization_options') or []) + (secondary.get('visualization_options') or [])
    primary['source'] = f"{primary.get('source', 'primary')}+{secondary.get('source', 'secondary')}"
    return primary


def _normalize_source_mode(mode):
    valid = {'live', 'database', 'both'}
    mode = (mode or 'both').strip().lower()
    return mode if mode in valid else 'both'


def get_modelsai_grounding_context(source_mode='both'):
    source_mode = _normalize_source_mode(source_mode)

    if source_mode == 'database':
        context = _dashboard_context_from_db() or {'source': 'database_unavailable', 'record_count': 0, 'columns_used': [], 'summary': {'note': 'Database source unavailable.'}, 'sample_rows': [], 'visualization_options': []}
        return _merge_recent_predictions_into_context(context)

    if source_mode == 'live':
        context = _dashboard_context_fallback()
        return _merge_recent_predictions_into_context(context)

    db_context = _dashboard_context_from_db()
    live_context = _dashboard_context_fallback()
    context = _combine_modelsai_contexts(db_context, live_context) if db_context and live_context else (db_context or live_context)
    return _merge_recent_predictions_into_context(context)


def _encode_current_plot_as_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def _next_modelsai_rotation_index(kind='chart'):
    global MODELSAI_CHART_ROTATION_INDEX
    global MODELSAI_STYLE_ROTATION_INDEX

    if kind == 'style':
        MODELSAI_STYLE_ROTATION_INDEX += 1
        return MODELSAI_STYLE_ROTATION_INDEX

    MODELSAI_CHART_ROTATION_INDEX += 1
    return MODELSAI_CHART_ROTATION_INDEX


def _modelsai_query_intents(user_query):
    q = (user_query or '').lower()
    intents = set()
    if any(k in q for k in ['shap', 'feature', 'driver', 'explain']):
        intents.add('explainability')
    if any(k in q for k in ['trend', 'trajectory', 'change', 'over time']):
        intents.add('trend')
    if any(k in q for k in ['compare', 'faculty', 'engineering', 'applied']):
        intents.add('faculty')
    if any(k in q for k in ['risk', 'fail', 'intervention']):
        intents.add('risk')
    if any(k in q for k in ['distribution', 'share', 'breakdown', 'mix']):
        intents.add('distribution')
    if not intents:
        intents.add('overview')
    return intents


def _select_diverse_chart_set(candidates, intents, max_charts=4):
    if not candidates:
        return []

    scored = []
    for item in candidates:
        tags = set(item.get('tags') or [])
        score = len(tags.intersection(intents)) * 10
        if 'overview' in tags:
            score += 1
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = [item for _, item in scored]
    shift = _next_modelsai_rotation_index('chart') % len(ordered)
    rotated = ordered[shift:] + ordered[:shift]

    picked = []
    picked_types = set()
    for item in rotated:
        if item.get('type') in picked_types:
            continue
        picked.append(item)
        picked_types.add(item.get('type'))
        if len(picked) >= max_charts:
            break

    if not picked:
        picked = rotated[:max_charts]

    signature = tuple(sorted(item.get('title', '') for item in picked))
    if MODELSAI_RECENT_CHART_SIGNATURES and signature == MODELSAI_RECENT_CHART_SIGNATURES[-1] and len(rotated) > len(picked):
        bumped = rotated[1:] + rotated[:1]
        picked = bumped[:max_charts]
        signature = tuple(sorted(item.get('title', '') for item in picked))

    MODELSAI_RECENT_CHART_SIGNATURES.append(signature)
    return picked


def build_modelsai_chart_images(context, user_query=None):
    """Build concrete chart images from grounded context summary data with variety across prompts."""
    chart_candidates = []
    summary = context.get('summary', {}) if isinstance(context, dict) else {}
    intents = _modelsai_query_intents(user_query)

    prediction_counts = summary.get('prediction_counts') or {}
    if prediction_counts:
        labels = list(prediction_counts.keys())
        values = [prediction_counts[k] for k in labels]
        colors = ['#28a745' if str(k).lower().startswith('pass') else '#ffc107' if str(k).lower().startswith('border') else '#dc3545' for k in labels]

        try:
            plt.figure(figsize=(7, 4))
            plt.bar(labels, values, color=colors)
            plt.title('Prediction Distribution')
            plt.xlabel('Prediction Class')
            plt.ylabel('Count')
            chart_candidates.append({'title': 'Prediction Distribution', 'type': 'bar', 'image': _encode_current_plot_as_base64(), 'tags': ['distribution', 'overview']})
        finally:
            plt.close('all')

        try:
            plt.figure(figsize=(6, 4))
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
            plt.title('Passing vs Borderline vs Failing')
            chart_candidates.append({'title': 'Prediction Class Share', 'type': 'pie', 'image': _encode_current_plot_as_base64(), 'tags': ['distribution', 'overview']})
        finally:
            plt.close('all')

    faculty_counts = summary.get('faculty_counts') or {}
    if faculty_counts:
        labels = list(faculty_counts.keys())
        values = [faculty_counts[k] for k in labels]
        try:
            plt.figure(figsize=(7, 4))
            plt.bar(labels, values, color=['#4e79a7', '#59a14f', '#f28e2b', '#e15759'][:len(labels)])
            plt.title('Predictions by Faculty')
            plt.xlabel('Faculty')
            plt.ylabel('Count')
            chart_candidates.append({'title': 'Predictions by Faculty', 'type': 'bar', 'image': _encode_current_plot_as_base64(), 'tags': ['faculty', 'overview']})
        finally:
            plt.close('all')

    prediction_by_faculty = summary.get('prediction_by_faculty') or {}
    if prediction_by_faculty:
        try:
            mix_df = pd.DataFrame(prediction_by_faculty).T.fillna(0)
            if not mix_df.empty:
                mix_df.plot(kind='bar', stacked=True, figsize=(8, 4), color=['#2a9d8f', '#e9c46a', '#e76f51', '#457b9d'])
                plt.title('Prediction Mix by Faculty')
                plt.xlabel('Faculty')
                plt.ylabel('Count')
                plt.legend(title='Prediction', bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout()
                chart_candidates.append({'title': 'Prediction Mix by Faculty', 'type': 'stacked_bar', 'image': _encode_current_plot_as_base64(), 'tags': ['faculty', 'distribution']})
        finally:
            plt.close('all')

    shap_top_features = summary.get('shap_top_features') or []
    if shap_top_features:
        try:
            shap_df = pd.DataFrame(shap_top_features).head(10)
            shap_df = shap_df.sort_values('importance', ascending=True)
            plt.figure(figsize=(8, 4.5))
            plt.barh(shap_df['feature'], shap_df['importance'], color='#2f6cad')
            plt.title('Top SHAP Feature Impact')
            plt.xlabel('Absolute SHAP impact (aggregate)')
            plt.ylabel('Feature')
            chart_candidates.append({'title': 'Top SHAP Feature Impact', 'type': 'barh', 'image': _encode_current_plot_as_base64(), 'tags': ['explainability', 'risk']})
        finally:
            plt.close('all')

    fail_probability_values = summary.get('fail_probability_values') or []
    if fail_probability_values:
        try:
            prob_series = pd.Series(fail_probability_values).dropna()
            if not prob_series.empty:
                plt.figure(figsize=(7, 4))
                plt.hist(prob_series, bins=10, color='#e15759', edgecolor='white')
                plt.title('Fail Probability Distribution')
                plt.xlabel('Fail probability')
                plt.ylabel('Students')
                chart_candidates.append({'title': 'Fail Probability Distribution', 'type': 'histogram', 'image': _encode_current_plot_as_base64(), 'tags': ['risk', 'distribution']})
        finally:
            plt.close('all')

        try:
            sorted_probs = sorted([float(v) for v in fail_probability_values])
            if sorted_probs:
                cumulative = np.cumsum(sorted_probs)
                plt.figure(figsize=(7, 4))
                plt.plot(range(1, len(cumulative) + 1), cumulative, color='#4e79a7', linewidth=2)
                plt.title('Cumulative Risk Curve')
                plt.xlabel('Student rank (low to high fail probability)')
                plt.ylabel('Cumulative fail probability')
                chart_candidates.append({'title': 'Cumulative Risk Curve', 'type': 'line', 'image': _encode_current_plot_as_base64(), 'tags': ['risk', 'trend']})
        finally:
            plt.close('all')

    faculty_avg_fail_probability = summary.get('faculty_avg_fail_probability') or {}
    if faculty_avg_fail_probability:
        try:
            labels = list(faculty_avg_fail_probability.keys())
            values = [faculty_avg_fail_probability[k] for k in labels]
            plt.figure(figsize=(7, 4))
            plt.bar(labels, values, color='#6c757d')
            plt.title('Average Fail Probability by Faculty')
            plt.xlabel('Faculty')
            plt.ylabel('Average fail probability')
            chart_candidates.append({'title': 'Average Fail Probability by Faculty', 'type': 'bar', 'image': _encode_current_plot_as_base64(), 'tags': ['faculty', 'risk']})
        finally:
            plt.close('all')

    selected = _select_diverse_chart_set(chart_candidates, intents, max_charts=4)
    return [{k: v for k, v in chart.items() if k != 'tags'} for chart in selected]


def build_modelsai_dashboard_payload(context):
    """Build compact dashboard payload for frontend interactive widgets."""
    summary = context.get('summary', {}) if isinstance(context, dict) else {}
    prediction_counts = summary.get('prediction_counts') or {}
    passing_count = int(prediction_counts.get('Passing', prediction_counts.get('Pass', 0)) or 0)
    failing_count = int(prediction_counts.get('Failing', prediction_counts.get('Fail', 0)) or 0)
    borderline_count = int(prediction_counts.get('Borderline', 0) or 0)

    cards = [
        {'label': 'Grounded Rows', 'value': int(context.get('record_count', 0) or 0)},
        {'label': 'Passing', 'value': passing_count},
        {'label': 'Borderline', 'value': borderline_count},
        {'label': 'Failing', 'value': failing_count},
        {'label': 'Avg Fail Probability', 'value': round(float(summary.get('avg_fail_probability') or 0.0), 3)},
        {'label': 'High Risk Count', 'value': int(summary.get('high_risk_count', 0) or 0)}
    ]

    return {
        'cards': cards,
        'faculty_counts': summary.get('faculty_counts') or {},
        'prediction_by_faculty': summary.get('prediction_by_faculty') or {},
        'top_shap_features': (summary.get('shap_top_features') or [])[:10],
        'top_predictions': (summary.get('top_risk_students') or [])[:10],
        'sample_rows': (context.get('sample_rows') or [])[:12]
    }


def _looks_like_markdown_table(block_lines):
    if len(block_lines) < 2:
        return False
    if '|' not in block_lines[0]:
        return False
    separator = block_lines[1].strip()
    return bool(re.match(r'^\|?\s*:?-{3,}', separator))


def _render_markdown_table(block_lines):
    headers = [html.escape(c.strip()) for c in block_lines[0].strip().strip('|').split('|')]
    rows = []
    for line in block_lines[2:]:
        if '|' not in line:
            continue
        rows.append([html.escape(c.strip()) for c in line.strip().strip('|').split('|')])

    thead = '<tr>' + ''.join([f'<th>{h}</th>' for h in headers]) + '</tr>'
    tbody = ''.join([
        '<tr>' + ''.join([f'<td>{cell}</td>' for cell in row]) + '</tr>'
        for row in rows
    ])
    return (
        "<div style='overflow-x:auto; margin: 10px 0;'>"
        "<table style='width:100%; border-collapse: collapse; font-size: 0.9rem;'>"
        f"<thead style='background:#e9ecef;'>{thead}</thead>"
        f"<tbody>{tbody}</tbody>"
        "</table></div>"
    )


def format_modelsai_response(response_text):
    """Render model markdown-like output into rich HTML (headings, lists, tables)."""
    if not response_text:
        return "<p>No response generated.</p>"

    lines = response_text.splitlines()
    blocks = []
    current = []
    for line in lines:
        if line.strip() == '':
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    rendered = ["<div style='font-family: Segoe UI, Arial, sans-serif; line-height: 1.6; color: #1f2933;'>"]

    for block in blocks:
        if _looks_like_markdown_table(block):
            rendered.append(_render_markdown_table(block))
            continue

        first = block[0].strip()
        if first.startswith('###'):
            rendered.append(f"<h5 style='margin: 14px 0 8px; color: #0f766e;'>{html.escape(first.lstrip('#').strip())}</h5>")
            if len(block) > 1:
                text = '<br>'.join([html.escape(l) for l in block[1:]])
                rendered.append(f"<p>{text}</p>")
            continue

        if first.startswith('##') or first.startswith('#'):
            rendered.append(f"<h4 style='margin: 14px 0 8px; color: #0f766e;'>{html.escape(first.lstrip('#').strip())}</h4>")
            if len(block) > 1:
                text = '<br>'.join([html.escape(l) for l in block[1:]])
                rendered.append(f"<p>{text}</p>")
            continue

        if all(l.strip().startswith(('- ', '* ')) for l in block):
            items = ''.join([f"<li>{html.escape(l.strip()[2:].strip())}</li>" for l in block])
            rendered.append(f"<ul style='margin: 8px 0 10px 18px;'>{items}</ul>")
            continue

        text = '<br>'.join([html.escape(l) for l in block])
        rendered.append(f"<p style='margin: 8px 0;'>{text}</p>")

    rendered.append('</div>')
    return ''.join(rendered)


def strip_visualization_boilerplate(response_text):
    """Remove repetitive visualization boilerplate so UI can offer concise follow-up suggestions instead."""
    if not response_text:
        return response_text

    lines = response_text.splitlines()
    cleaned = []
    skip = False
    for line in lines:
        lower = line.strip().lower()

        if lower.startswith('visualization options'):
            skip = True
            continue

        if skip:
            if lower.startswith('#') or lower.startswith('executive') or lower.startswith('recommended') or lower.startswith('summary'):
                skip = False
                cleaned.append(line)
            elif lower == '':
                continue
            else:
                continue
        else:
            cleaned.append(line)

    return '\n'.join(cleaned).strip()


def _next_modelsai_style_hint():
    styles = [
        'Use an executive briefing tone with short, punchy insights.',
        'Use a diagnostic tone focused on root-cause and drivers.',
        'Use an intervention playbook tone with clear priority actions.',
        'Use a comparative tone highlighting contrasts between groups.',
        'Use a narrative analyst tone that explains what changed and why it matters.'
    ]
    idx = _next_modelsai_rotation_index('style') % len(styles)
    return styles[idx]


def build_modelsai_system_prompt(context, user_query=None, style_hint=None):
    context_json = json.dumps(context, default=str)
    user_query = user_query or ''
    style_hint = style_hint or _next_modelsai_style_hint()
    return f"""
You are ModelsAI, an advanced analytics assistant for Durban University of Technology.

You must ground answers in the provided dashboard context JSON. If data is missing, say so clearly.

Current user request:
{user_query}

Dashboard context JSON:
{context_json}

Narrative lens for this response:
{style_hint}

Response rules:
1. Use clear markdown sections with headings.
2. Include concise executive insights and concrete actions.
3. Use markdown tables whenever comparing groups or metrics.
4. Do not include a long "Visualization Options" section by default.
5. If the user explicitly asks for plots, mention at most 2 concise chart suggestions.
5. Ground findings in prediction counts, top-risk students, and SHAP feature impact when available.
6. Explain which features (from SHAP) are driving risk and what interventions correspond to those features.
7. Keep recommendations practical for academic support and early intervention.
8. Do not repeat prior wording patterns; vary phrasing and structure while staying factual.
"""


@app.route('/api/modelsai/settings', methods=['GET'])
def modelsai_settings():
    settings = get_modelsai_settings()
    return jsonify({
        'provider': settings['provider'],
        'temperature': settings['temperature'],
        'max_tokens': settings['max_tokens'],
        'openai_model': settings['openai_model'],
        'config_file': 'backend/.env',
        'available_providers': ['openai']
    }), 200


@app.route('/api/modelsai/context', methods=['GET'])
def modelsai_context():
    """Return grounded context snapshot for frontend widgets without LLM latency."""
    source_mode = _normalize_source_mode(request.args.get('source_mode', 'both'))
    grounding_context = get_modelsai_grounding_context(source_mode=source_mode)
    chart_images = build_modelsai_chart_images(grounding_context, user_query='overview dashboard context')
    dashboard_data = build_modelsai_dashboard_payload(grounding_context)
    return jsonify({
        'source_mode': source_mode,
        'grounding_source': grounding_context.get('source'),
        'visualization_options': grounding_context.get('visualization_options', []),
        'chart_images': chart_images,
        'dashboard_data': dashboard_data
    }), 200

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/dashsite')
def dashsite():
    return render_template('dashsite.html')

@app.route('/flagged-students')
def flagged_students():
    return render_template('flagged_students.html')

@app.route('/api/recommend-flagged', methods=['POST'])
def recommend_flagged():
    """Auto-generate recommendations for a flagged student."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        student_data = data.get('student_data', {})
        faculty = data.get('faculty', 'unknown')
        
        print(f"\n[RECOMMEND] Request received for faculty={faculty}, data keys: {list(student_data.keys())}")

        # Map age group string to approximate numeric age
        age_group_map = {
            'Teenager': 17, '15-19': 17, '20-29': 24,
            '30-39': 34, '40-49': 44, '>50': 52
        }
        age = age_group_map.get(str(student_data.get('AgeGroup', '20-29')), 24)

        # Map ResidenceIndicator → Residence Flag (Y/N)
        res_raw = student_data.get('ResidenceIndicator', 'No')
        residence_flag = 'Y' if str(res_raw).lower() in ['yes', 'y', 'true'] else 'N'

        student_info = {
            'Age': age,
            'Gender': student_data.get('Gender', 'M'),
            'Previous Activity Desc': student_data.get('PreviousActivity', 'SCHOOL'),
            'Language Name': student_data.get('Language', 'ENGLISH'),
            'Ethnic Group Name': student_data.get('Race', 'AFRICAN'),
            'Church Religion Name': student_data.get('Religion', 'CHRISTIAN'),
            'BRICS Country': 'South Africa',
            'Race Equity': student_data.get('Race', 'AFRICAN'),
            'Disability Flag': 'N',
            'Student Type Description': student_data.get('StudentTypeDescription', 'NORMAL STUDENT'),
            'Residence Flag': residence_flag,
            'Bursary Flag': 'N',
            'previous_mark': float(student_data.get('FinalYearSymbol', 50)),
            'stats_credit': float(student_data.get('Quintile', 3)) * 20,
            'preferred_faculty': faculty,
            'completed_courses': set()
        }
        
        print(f"[RECOMMEND] Mapped student info: Age={age}, Gender={student_info['Gender']}, Residence={residence_flag}")
        print(f"[RECOMMEND] Calling recommender.recommend_based_on_similar_students()...")

        recommendations = recommender.recommend_based_on_similar_students(student_info)
        
        # Convert numpy types to native Python types for JSON serialization
        recommendations_clean = json.loads(json.dumps(recommendations, cls=NumpyEncoder))
        
        print(f"[RECOMMEND] ✓ Recommendations generated successfully")
        return jsonify(recommendations_clean), 200

    except Exception as e:
        print(f"[RECOMMEND] ✗ Error: {str(e)}")
        print(f"[RECOMMEND] Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/engineering')
def engineering():
    return render_template('engineering.html')

@app.route('/applied-sciences')
def applied_sciences():
    return render_template('applied.html')

@app.route('/recommend', methods=['GET'])
def input_values():
    return render_template('recommend.html')

@app.route('/results', methods=['POST'])
def results():
    student_info = {
        'Age': int(request.form['age']),
        'Gender': request.form['gender'],
        'Previous Activity Desc': request.form['previous_activity_desc'],
        'Language Name': request.form['language_name'],
        'Ethnic Group Name': request.form['ethnic_group_name'],
        'Church Religion Name': request.form['church_religion_name'],
        'BRICS Country': request.form['brics_country'],
        'Race Equity': request.form['race_equity'],
        'Disability Flag': request.form['disability_flag'],
        'Student Type Description': request.form['student_type_description'],
        'Residence Flag': request.form['residence_flag'],
        'Bursary Flag': request.form['bursary_flag'],
        'previous_mark': float(request.form['previous_mark']),
        'stats_credit': float(request.form['stats_credit']),
        'completed_courses': set(request.form['completed_courses'].split(','))
    }
    recommendations = recommender.recommend_based_on_similar_students(student_info)
    return render_template('results.html', recommendations=recommendations)

@app.route('/predictions/engineering', methods=['POST'])
def engineering_prediction():
    try:
        input_data = request.json
        print("Raw input data:", input_data)
        
        # Validate exam_month range
        exam_month = int(input_data.get('exam_month'))
        if exam_month < 1 or exam_month > 12:
            raise ValueError("Exam Month must be between 1 and 12")
        
        # Map frontend keys to model feature names
        input_df = pd.DataFrame([{
            'Residence': input_data.get('residence'),
            'FTEN Status': input_data.get('ften_status'),
            'Bursary': input_data.get('bursary'),
            'Offering Type': input_data.get('offering_type'),
            'Disability': input_data.get('disability'),
            'Age Group': input_data.get('age_group'),
            'Exam Type': input_data.get('exam_type'),
            'Exam Month': exam_month,
            'Race': input_data.get('race'),
            'Gender': input_data.get('gender'),
            'Stud_Type': input_data.get('stud_type'),
            'Marital Status': input_data.get('marital_status')
        }])
        
        print("DataFrame before encoding:", input_df)
        
        # Validate all required features are present
        missing_features = set(engineering_features) - set(input_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Encode categorical features
        for col in engineering_categories.keys():
            if col in input_df.columns:
                le = LabelEncoder()
                allowed_values = engineering_categories[col]
                le.fit(allowed_values)
                safe_values = input_df[col].apply(lambda x: _match_allowed_value(x, allowed_values)).astype(str)
                input_df[col] = le.transform(safe_values)
        
        # Ensure feature order matches scaler
        try:
            scaler_feature_order = engineering_scaler.feature_names_in_
        except AttributeError:
            scaler_feature_order = engineering_features  # Fallback if feature_names_in_ is not available
        input_df = input_df[scaler_feature_order]
        print("Scaler input DataFrame:", input_df)
        print("Scaler expected features:", scaler_feature_order)
        
        # Scale all features
        scaled_features = engineering_scaler.transform(input_df)
        input_df = pd.DataFrame(scaled_features, columns=scaler_feature_order)
        
        # Reorder columns to match model training feature order
        input_df = input_df[engineering_features]
        print("Final features being sent to model:", input_df.columns.tolist())
        
        # Verify feature names match model expectations
        expected_features = engineering_model.get_booster().feature_names
        if input_df.columns.tolist() != expected_features:
            # Handle missing or extra features
            input_df_encoded = pd.DataFrame(0, index=input_df.index, columns=expected_features)
            for col in input_df.columns:
                if col in expected_features:
                    input_df_encoded[col] = input_df[col]
            input_df = input_df_encoded
            print("Adjusted features after encoding:", input_df.columns.tolist())
        
        # Make prediction
        prediction = engineering_model.predict(input_df)
        probabilities = engineering_model.predict_proba(input_df)
        
        # SHAP explanation
        explainer = shap.TreeExplainer(engineering_model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (positive class) for binary classification
        
        shap_plot = create_shap_plot(shap_values[0], input_df.columns)

        event = {
            'student_id': input_data.get('student_number') or input_data.get('student_id') or f"eng-{int(datetime.utcnow().timestamp())}",
            'faculty': 'engineering',
            'result': "Passing" if prediction[0] == 1 else "Failing",
            'pass_probability': float(probabilities[0][1]),
            'fail_probability': float(probabilities[0][0]),
            'confidence': float(abs(float(probabilities[0][1]) - float(probabilities[0][0]))),
            'intervention_needed': 'Yes' if float(probabilities[0][1]) < 0.5 else 'No',
            'shap_values': {feature: float(value) for feature, value in zip(input_df.columns, shap_values[0])},
            'features': {k: input_data.get(k) for k in input_data.keys()} if isinstance(input_data, dict) else {}
        }
        _register_modelsai_prediction_event(event)
        
        return jsonify({
            'prediction': {
                'result': "Student Passed" if prediction[0] == 1 else "Student Failed",
                'pass_probability': float(probabilities[0][1]),
                'fail_probability': float(probabilities[0][0])
            },
            'shap_plot': shap_plot,
            'shap_values': {feature: float(value) for feature, value in zip(input_df.columns, shap_values[0])},
            'feature_names': input_df.columns.tolist()  # Include feature names for frontend validation
        }), 200
        
    except Exception as e:
        print(f"Engineering prediction error: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return jsonify({
            'error': str(e),
            'expected_features': engineering_model.get_booster().feature_names,
            'received_features': input_df.columns.tolist() if 'input_df' in locals() else []
        }), 500

@app.route('/predictions/applied-sciences', methods=['POST'])
def applied_sciences_prediction():
    try:
        input_data = request.json
        print("Raw input data:", input_data)
        
        # Validate exam_month range
        exam_month = int(input_data.get('exam_month'))
        if exam_month < 1 or exam_month > 12:
            raise ValueError("Exam Month must be between 1 and 12")
        
        # Map frontend keys to model feature names
        input_df = pd.DataFrame([{
            'Exam Type': input_data.get('exam_type'),
            'Gender': input_data.get('gender'),
            'Race': input_data.get('race'),
            'Disability': input_data.get('disability'),
            'Department': input_data.get('department'),
            'Final Year': input_data.get('final_year'),
            'FTEN Status': input_data.get('ften_status'),
            'Residence': input_data.get('residence'),
            'Marital Status': input_data.get('marital_status'),
            'Exam Month': exam_month,
            'Age Group': input_data.get('age_group'),
            'Stud Type': input_data.get('stud_type'),
            'Bursary': input_data.get('bursary'),
            'Offer Type': input_data.get('offer_type')
        }])
        
        print("DataFrame before encoding:", input_df)
        
        # Validate all required features are present
        missing_features = set(applied_features) - set(input_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Encode categorical features
        for col in applied_categories.keys():
            if col in input_df.columns:
                le = LabelEncoder()
                allowed_values = applied_categories[col]
                le.fit(allowed_values)
                safe_values = input_df[col].apply(lambda x: _match_allowed_value(x, allowed_values)).astype(str)
                input_df[col] = le.transform(safe_values)
        
        # Ensure feature order matches scaler
        try:
            scaler_feature_order = applied_scaler.feature_names_in_
        except AttributeError:
            scaler_feature_order = applied_features  # Fallback if feature_names_in_ is not available
        input_df = input_df[scaler_feature_order]
        print("Scaler input DataFrame:", input_df)
        print("Scaler expected features:", scaler_feature_order)
        
        # Scale all features
        scaled_features = applied_scaler.transform(input_df)
        input_df = pd.DataFrame(scaled_features, columns=scaler_feature_order)
        
        # Reorder columns to match model training feature order
        input_df = input_df[applied_features]
        print("Final features being sent to model:", input_df.columns.tolist())
        
        # Verify feature names match model expectations
        expected_features = applied_model.get_booster().feature_names
        if input_df.columns.tolist() != expected_features:
            raise ValueError(
                f"Feature names mismatch.\n"
                f"Expected: {expected_features}\n"
                f"Received: {input_df.columns.tolist()}"
            )
        
        # Make prediction
        prediction = applied_model.predict(input_df)
        probabilities = applied_model.predict_proba(input_df)
        
        # SHAP explanation
        explainer = shap.TreeExplainer(applied_model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use class 1 (positive class) for binary classification
        
        # Ensure SHAP values align with feature names
        shap_plot = create_shap_plot(shap_values[0], applied_features)
        
        # Create SHAP values dictionary with explicit feature mapping
        shap_values_dict = {feature: float(value) for feature, value in zip(applied_features, shap_values[0])}

        event = {
            'student_id': input_data.get('student_number') or input_data.get('student_id') or f"app-{int(datetime.utcnow().timestamp())}",
            'faculty': 'applied_sciences',
            'result': "Passing" if prediction[0] == 1 else "Failing",
            'pass_probability': float(probabilities[0][1]),
            'fail_probability': float(probabilities[0][0]),
            'confidence': float(abs(float(probabilities[0][1]) - float(probabilities[0][0]))),
            'intervention_needed': 'Yes' if float(probabilities[0][1]) < 0.5 else 'No',
            'shap_values': shap_values_dict,
            'features': {k: input_data.get(k) for k in input_data.keys()} if isinstance(input_data, dict) else {}
        }
        _register_modelsai_prediction_event(event)
        
        return jsonify({
            'prediction': {
                'result': "Student Passed" if prediction[0] == 1 else "Student Failed",
                'pass_probability': float(probabilities[0][1]),
                'fail_probability': float(probabilities[0][0])
            },
            'shap_plot': shap_plot,
            'shap_values': shap_values_dict,
            'feature_names': applied_features  # Include feature names for frontend validation
        }), 200
        
    except Exception as e:
        print(f"Applied sciences prediction error: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return jsonify({
            'error': str(e),
            'expected_features': applied_model.get_booster().feature_names,
            'received_features': input_df.columns.tolist() if 'input_df' in locals() else []
        }), 500

@app.route('/ai')
def ai():
    return render_template('ai.html')

@app.route('/api/ai', methods=['POST'])
def ai_response():
    try:
        client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=os.getenv('OPENAI_API_VERSION'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
        )
        
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOY_NAME'),
            messages=[
                {"role": "system", "content": "You are DUTAi, a helpful AI assistant for Durban University of Technology students."},
                {"role": "user", "content": request.json.get('message')}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return jsonify({
            'response': format_ai_response(response.choices[0].message.content)
        }), 200
    except Exception as e:
        print(f"AI response error: {e}")
        return jsonify({'error': 'Failed to generate response'}), 500

@app.route('/modelsai')
def modelsai():
    return render_template('modelsai.html')

@app.route('/api/modelsai', methods=['POST'])
def modelsai_response():
    try:
        payload = request.json or {}
        message = (payload.get('message') or '').strip()
        source_mode = _normalize_source_mode(payload.get('source_mode', 'both'))
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        if _is_small_talk_message(message):
            return jsonify({
                'response': _small_talk_reply(),
                'provider': 'local-fastpath',
                'model': 'none',
                'source_mode': source_mode,
                'grounding_source': 'none',
                'visualization_options': [],
                'chart_images': []
            }), 200

        settings = get_modelsai_settings()
        provider = 'openai'

        client, model_name = create_modelsai_client(provider)
        grounding_context = get_modelsai_grounding_context(source_mode=source_mode)
        chart_images = build_modelsai_chart_images(grounding_context, user_query=message)
        dashboard_data = build_modelsai_dashboard_payload(grounding_context)
        style_hint = _next_modelsai_style_hint()
        system_prompt = build_modelsai_system_prompt(grounding_context, user_query=message, style_hint=style_hint)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=settings['max_tokens'],
            temperature=settings['temperature']
        )

        raw_text = response.choices[0].message.content
        raw_text = strip_visualization_boilerplate(raw_text)
        
        return jsonify({
            'response': format_modelsai_response(raw_text),
            'provider': provider,
            'model': model_name,
            'source_mode': source_mode,
            'grounding_source': grounding_context.get('source'),
            'visualization_options': grounding_context.get('visualization_options', []),
            'chart_images': chart_images,
            'dashboard_data': dashboard_data,
            'response_style': style_hint
        }), 200
    except Exception as e:
        print(f"ModelsAI error: {e}")
        return jsonify({'error': 'Failed to generate response', 'details': str(e)}), 500

@app.route('/api/modelsai-query', methods=['POST'])
def modelsai_query():
    try:
        nl_query = request.json.get('message').lower().strip()
        schema = get_database_schema()
        
        if not schema:
            return jsonify({'error': 'Database schema unavailable'}), 500
        
        # Handle metadata queries
        if 'how many tables' in nl_query:
            count = sum(1 for obj in schema.values() if obj['type'] == 'USER_TABLE')
            return jsonify({'response': f"There are {count} tables."})
        
        if 'list tables' in nl_query:
            tables = [name for name, obj in schema.items() if obj['type'] == 'USER_TABLE']
            return jsonify({'response': f"Tables: {', '.join(tables)}"})
        
        if 'list views' in nl_query:
            views = [name for name, obj in schema.items() if obj['type'] == 'VIEW']
            return jsonify({'response': f"Views: {', '.join(views)}"})
        
        # Generate and execute SQL
        sql = generate_sql(nl_query, schema)
        df = execute_query(sql)
        
        if df is None:
            return jsonify({'error': 'Query failed'}), 500
        if df.empty:
            return jsonify({'response': "No results found."})
        
        # Create visualization
        plot = None
        if len(df.columns) >= 1:
            x_col = df.columns[0]
            y_col = df.columns[1] if len(df.columns) >= 2 else None
            plot_type = 'bar' if len(df) < 20 else 'line'
            plot = create_data_plot(df, x_col, y_col, plot_type)
        
        return jsonify({
            'response': analyze_data(df),
            'plot': plot,
            'data_preview': df.head().to_dict('records')
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({'error': f"Processing failed: {str(e)}"}), 500

@app.route('/api/schema', methods=['GET'])
def get_schema():
    schema = get_database_schema()
    return jsonify({'schema': schema}) if schema else jsonify({'error': 'Schema unavailable'}), 500

# Cleanup handler
@app.teardown_request
def cleanup(_):
    plt.close('all')

if __name__ == '__main__':
    app.run(host='0.0.0.0', 
            port=int(os.getenv('PORT', 5001)), 
            debug=os.getenv('DEBUG') == 'True')