# Set matplotlib backend first (MUST BE FIRST IMPORT)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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
import html
from joblib import load

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
applied_model = joblib.load('xgboost_applied_base.pkl')
engineering_model = joblib.load('xgboost_model_cv.pkl')
applied_scaler = joblib.load('scaler_applied.pkl')
engineering_scaler = joblib.load('scaler_cv.pkl')

# Load dataset for recommendations
with open('combined_student_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Initialize recommender
recommender = EnhancedRecommender()
recommender.load_data(data)

# Define categorical columns
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

# Database connection
def get_db_connection():
    connection_string = os.getenv('CONNECTION_STRING')
    try:
        return pyodbc.connect(connection_string)
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
    return {
        'provider': 'openai',
        'temperature': float(os.getenv('MODELSAI_TEMPERATURE', '0.35')),
        'max_tokens': int(os.getenv('MODELSAI_MAX_TOKENS', '1400')),
        'openai_model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    }


def create_modelsai_client(provider):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY is missing')
    client = OpenAI(api_key=api_key, base_url=os.getenv('OPENAI_BASE_URL') or None)
    return client, os.getenv('OPENAI_MODEL', 'gpt-4o-mini')


def _safe_column(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match:
            return match
    return None


def get_modelsai_grounding_context():
    schema = get_database_schema() or {}
    view_name = None
    for obj_name in schema.keys():
        if 'stud_ml_predictions' in obj_name.lower():
            view_name = obj_name
            break

    if view_name:
        columns = [c['name'] for c in schema[view_name]['columns']]
        pred_col = _safe_column(columns, ['Prediction'])
        faculty_col = _safe_column(columns, ['Faculty', 'Faculty_Name', 'Faculty Name'])
        fail_prob_col = _safe_column(columns, ['Probability_Fail', 'Fail_Probability'])

        select_cols = [c for c in [pred_col, faculty_col, fail_prob_col] if c]
        if select_cols:
            schema_name, object_name = view_name.split('.', 1)
            select_sql = ', '.join([f'[{c}]' for c in select_cols])
            df = execute_query(f"SELECT TOP 2000 {select_sql} FROM [{schema_name}].[{object_name}]")
            if df is not None and not df.empty:
                context = {
                    'source': view_name,
                    'record_count': int(len(df)),
                    'columns_used': select_cols,
                    'sample_rows': df.head(20).fillna('').to_dict('records'),
                    'summary': {},
                    'visualization_options': []
                }
                if pred_col and pred_col in df.columns:
                    context['summary']['prediction_counts'] = df[pred_col].value_counts(dropna=False).to_dict()
                    context['visualization_options'].append({'type': 'bar', 'title': 'Prediction distribution', 'x': pred_col, 'y': 'Count'})
                if faculty_col and faculty_col in df.columns:
                    context['summary']['faculty_counts'] = df[faculty_col].value_counts(dropna=False).to_dict()
                    context['visualization_options'].append({'type': 'bar', 'title': 'Predictions by faculty', 'x': faculty_col, 'y': 'Count'})
                if fail_prob_col and fail_prob_col in df.columns:
                    numeric = pd.to_numeric(df[fail_prob_col], errors='coerce')
                    context['summary']['avg_fail_probability'] = float(numeric.mean()) if numeric.notna().any() else None
                    context['visualization_options'].append({'type': 'histogram', 'title': 'Fail probability distribution', 'x': fail_prob_col, 'y': 'Frequency'})
                return context

    fallback_df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    return {
        'source': 'combined_student_data.pkl',
        'record_count': int(len(fallback_df)) if not fallback_df.empty else 0,
        'columns_used': list(fallback_df.columns) if not fallback_df.empty else [],
        'sample_rows': fallback_df.head(20).fillna('').to_dict('records') if not fallback_df.empty else [],
        'summary': {'note': 'Using local fallback dataset context.'},
        'visualization_options': []
    }


def format_modelsai_response(response_text):
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
        if len(block) >= 2 and '|' in block[0] and re.match(r'^\|?\s*:?-{3,}', block[1].strip()):
            headers = [html.escape(c.strip()) for c in block[0].strip().strip('|').split('|')]
            rows = []
            for line in block[2:]:
                if '|' in line:
                    rows.append([html.escape(c.strip()) for c in line.strip().strip('|').split('|')])
            thead = '<tr>' + ''.join([f'<th>{h}</th>' for h in headers]) + '</tr>'
            tbody = ''.join(['<tr>' + ''.join([f'<td>{cell}</td>' for cell in row]) + '</tr>' for row in rows])
            rendered.append("<div style='overflow-x:auto; margin: 10px 0;'><table style='width:100%; border-collapse: collapse; font-size: 0.9rem;'><thead style='background:#e9ecef;'>" + thead + "</thead><tbody>" + tbody + "</tbody></table></div>")
            continue

        first = block[0].strip()
        if first.startswith('#'):
            level = 'h5' if first.startswith('###') else 'h4'
            rendered.append(f"<{level} style='margin: 14px 0 8px; color: #0f766e;'>{html.escape(first.lstrip('#').strip())}</{level}>")
            if len(block) > 1:
                rendered.append(f"<p>{'<br>'.join([html.escape(l) for l in block[1:]])}</p>")
            continue

        if all(l.strip().startswith(('- ', '* ')) for l in block):
            items = ''.join([f"<li>{html.escape(l.strip()[2:].strip())}</li>" for l in block])
            rendered.append(f"<ul style='margin: 8px 0 10px 18px;'>{items}</ul>")
            continue

        rendered.append(f"<p style='margin: 8px 0;'>{'<br>'.join([html.escape(l) for l in block])}</p>")

    rendered.append('</div>')
    return ''.join(rendered)


def build_modelsai_system_prompt(context):
    context_json = json.dumps(context, default=str)
    return f"""
You are ModelsAI, an advanced analytics assistant for Durban University of Technology.
Ground every response in the provided dashboard context JSON and explicitly call out data gaps.

Dashboard context JSON:
{context_json}

Response requirements:
1. Use markdown headings and concise sections.
2. Add markdown tables for metric comparisons.
3. Include a section named "Visualization Options" with chart/graph ideas when relevant.
4. Provide practical intervention-oriented insights.
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

# Routes
@app.route('/')
def home():
    return render_template('home.html')

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

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/predictions/engineering', methods=['POST'])
def engineering_prediction():
    try:
        input_data = request.json
        input_df = pd.DataFrame([{
            'Residence': input_data.get('residence'),
            'FTEN Status': input_data.get('ften_status'),
            'Bursary': input_data.get('bursary'),
            'Offering Type': input_data.get('offering_type'),
            'Disability': input_data.get('disability'),
            'Age Group': input_data.get('age_group'),
            'Exam Type': input_data.get('exam_type'),
            'Exam Month': int(input_data.get('exam_month')),
            'Race': input_data.get('race'),
            'Gender': input_data.get('gender'),
            'Stud_Type': input_data.get('stud_type'),
            'Marital Status': input_data.get('marital_status')
        }])
        
        # Encode categorical features
        for col, values in engineering_categories.items():
            le = LabelEncoder()
            le.fit(values)
            input_df[col] = le.transform(input_df[col].astype(str))
        
        # Scale numerical features
        input_df[['Exam Month']] = engineering_scaler.transform(input_df[['Exam Month']])
        
        # Ensure correct column order
        input_df = input_df[[
            'Residence', 'FTEN Status', 'Bursary', 'Offering Type', 'Disability', 'Age Group',
            'Exam Type', 'Exam Month', 'Race', 'Gender', 'Stud_Type', 'Marital Status'
        ]]
        
        # Make prediction
        prediction = engineering_model.predict(input_df)
        probabilities = engineering_model.predict_proba(input_df)
        
        # SHAP explanation
        explainer = shap.TreeExplainer(engineering_model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Print SHAP values for debugging
        print(f"SHAP values: {shap_values}")
        
        # Create SHAP plot
        shap_plot = create_shap_plot(shap_values[0], input_df.columns)
        
        return jsonify({
            'prediction': {
                'result': "Student Passed" if prediction[0] == 1 else "Student Failed",
                'pass_probability': float(probabilities[0][1]),
                'fail_probability': float(probabilities[0][0])
            },
            'shap_plot': shap_plot,
            'shap_values': {feature: float(value) for feature, value in zip(input_df.columns, shap_values[0])}
        }), 200
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predictions/applied-sciences', methods=['POST'])
def applied_sciences_prediction():
    try:
        input_data = request.json
        input_df = pd.DataFrame([{
            'Department': input_data.get('department'),
            'Marital Status': input_data.get('marital_status'),
            'Gender': input_data.get('gender'),
            'Race': input_data.get('race'),
            'Disability': input_data.get('disability'),
            'Stud Type': input_data.get('stud_type'),
            'Exam Month': int(input_data.get('exam_month')),
            'Exam Type': input_data.get('exam_type'),
            'Offer Type': input_data.get('offer_type'),
            'FTEN Status': input_data.get('ften_status'),
            'Final Year': input_data.get('final_year'),
            'Residence': input_data.get('residence'),
            'Bursary': input_data.get('bursary'),
            'Age Group': input_data.get('age_group')
        }])
        
        # Encode categorical features
        for col, values in applied_categories.items():
            le = LabelEncoder()
            le.fit(values)
            input_df[col] = le.transform(input_df[col].astype(str))
        
        # Scale numerical features
        input_df[['Exam Month']] = applied_scaler.transform(input_df[['Exam Month']])
        
        # Ensure correct column order
        input_df = input_df[[
            'Exam Type', 'Gender', 'Race', 'Disability', 'Department', 'Final Year',
            'FTEN Status', 'Residence', 'Marital Status', 'Exam Month',
            'Age Group', 'Stud Type', 'Bursary', 'Offer Type'
        ]]
        
        # Make prediction
        prediction = applied_model.predict(input_df)
        probabilities = applied_model.predict_proba(input_df)
        
        # SHAP explanation
        explainer = shap.TreeExplainer(applied_model)
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_plot = create_shap_plot(shap_values[0], input_df.columns)
        
        return jsonify({
            'prediction': {
                'result': "Student Passed" if prediction[0] == 1 else "Student Failed",
                'pass_probability': float(probabilities[0][1]),
                'fail_probability': float(probabilities[0][0])
            },
            'shap_plot': shap_plot,
            'shap_values': {feature: float(value) for feature, value in zip(input_df.columns, shap_values[0])}
        }), 200
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

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
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        settings = get_modelsai_settings()
        provider = 'openai'

        client, model_name = create_modelsai_client(provider)
        grounding_context = get_modelsai_grounding_context()
        system_prompt = build_modelsai_system_prompt(grounding_context)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=settings['max_tokens'],
            temperature=settings['temperature']
        )
        
        return jsonify({
            'response': format_modelsai_response(response.choices[0].message.content),
            'provider': provider,
            'model': model_name,
            'grounding_source': grounding_context.get('source'),
            'visualization_options': grounding_context.get('visualization_options', [])
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