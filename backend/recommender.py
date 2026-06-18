# recommender.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
#from services.recommender import EnhancedRecommender
import numpy as np

class EnhancedRecommender:
    def __init__(self):
        self.data = None
        self.student_features = None
        self.course_features = None
        self.faculty_features = None
        self.department_features = None
        self.student_scaler = StandardScaler()
        self.student_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.course_scaler = StandardScaler()
        self.faculty_scaler = StandardScaler()
        self.department_scaler = StandardScaler()
        self.course_codes = None
        self.faculty_names = None
        self.department_names = None
        self.course_mapping = None

    def load_data(self, data):
        """Load and preprocess the student data"""
        self.data = data
        self._prepare_features()

    def _prepare_features(self):
        """Prepare feature matrices for students, courses, faculties, and departments"""
        # Ensure numeric columns are of the correct type
        self.data['Final Mark'] = pd.to_numeric(self.data['Final Mark'], errors='coerce')
        self.data['Stats Credit'] = pd.to_numeric(self.data['Stats Credit'], errors='coerce')
        self.data['Age'] = pd.to_numeric(self.data['Age'], errors='coerce')

        # Fill missing values
        self.data['Final Mark'] = self.data['Final Mark'].fillna(0)
        self.data['Stats Credit'] = self.data['Stats Credit'].fillna(0)
        self.data['Age'] = self.data['Age'].fillna(0)

        # Student Features
        numerical_features = self.data[['Age', 'Final Mark', 'Stats Credit']]
        categorical_features = self.data[['Gender', 'Previous Activity Desc', 'Language Name', 'Ethnic Group Name',
                                          'Church Religion Name', 'BRICS Country', 'Race Equity',
                                          'Disability Flag', 'Student Type Description',
                                          'Residence Flag', 'Bursary Flag']]

        # Scale numerical features
        scaled_numerical = self.student_scaler.fit_transform(numerical_features)

        # Encode categorical features
        encoded_categorical = self.student_encoder.fit_transform(categorical_features)

        # Combine student features
        self.student_features = np.hstack([scaled_numerical, encoded_categorical])

        # Course Features
        self._prepare_course_features()

        # Faculty Features
        self._prepare_faculty_features()

        # Department Features
        self._prepare_department_features()

    def _prepare_course_features(self):
        """Prepare features for courses"""
        if 'Subject Code' in self.data.columns:
            self.course_features = self.data[['Subject Code']].drop_duplicates().reset_index(drop=True)
        else:
            self.course_features = pd.DataFrame()

    def _prepare_faculty_features(self):
        """Prepare features for faculties"""
        if 'Faculty Name' in self.data.columns:
            self.faculty_features = self.data[['Faculty Name']].drop_duplicates().reset_index(drop=True)
        else:
            self.faculty_features = pd.DataFrame()

    def _prepare_department_features(self):
        """Prepare features for departments"""
        if 'Department' in self.data.columns:
            self.department_features = self.data[['Department']].drop_duplicates().reset_index(drop=True)
        else:
            self.department_features = pd.DataFrame()

    def _prepare_query_features(self, student_info):
        """Prepare features for a query student"""
        numerical = pd.DataFrame({
           'Age': [student_info['Age']],
           'Final Mark': [student_info.get('previous_mark', 0)],
           'Stats Credit': [student_info.get('stats_credit', 0)]
        })
        categorical = np.array([[
            student_info['Gender'],
            student_info['Previous Activity Desc'],
            student_info['Language Name'],
            student_info['Ethnic Group Name'],
            student_info['Church Religion Name'],
            student_info['BRICS Country'],
            student_info['Race Equity'],
            student_info['Disability Flag'],
            student_info['Student Type Description'],
            student_info['Residence Flag'],
            student_info['Bursary Flag']
        ]])
        encoded_categorical = self.student_encoder.transform(pd.DataFrame(categorical, columns=[
            'Gender', 'Previous Activity Desc', 'Language Name', 'Ethnic Group Name', 'Church Religion Name',
            'BRICS Country', 'Race Equity', 'Disability Flag', 'Student Type Description',
            'Residence Flag', 'Bursary Flag'
        ]))
        return np.hstack([
            self.student_scaler.transform(numerical),
            encoded_categorical
        ])

    def calculate_pass_rate(self, group_data):
        """Calculate pass rate for a group of data"""
        if group_data.empty:
            return 0.0
        pass_count = (group_data['Pass Fail'] == 'P').sum()
        total_count = len(group_data)
        return round((pass_count / total_count) * 100,1)if total_count > 0 else 0.0

    def get_similar_students(self, student_info, n_similar=5):
        """Find similar students using cosine similarity"""
        query_features = self._prepare_query_features(student_info)
        similarities = cosine_similarity(query_features.reshape(1, -1), self.student_features)
        similar_indices = similarities[0].argsort()[::-1]

        similar_students = []
        seen_ids = set()

        for idx in similar_indices:
            student_data = self.data.iloc[idx]
            student_id = student_data['Student Number']
            if student_id in seen_ids:
                continue
            seen_ids.add(student_id)

            similar_students.append({
                'student_id': student_id,
                'similarity_score': round(similarities[0][idx], 3),
                'faculty': student_data['Faculty Name'],
                'department': student_data['Department'],
                'qualification': student_data.get('Qualification Code', 'N/A'),
                'performance': round(student_data['Final Mark'], 2),
                'pass_status': student_data['Pass Fail']
            })

            if len(similar_students) >= n_similar:
                break

        return similar_students

    def recommend_based_on_similar_students(self, student_info, n_recommendations=5):
        """Provide personalised recommendations based on the student's academic profile."""
        previous_mark = float(student_info.get('previous_mark', 0))
        stats_credit  = float(student_info.get('stats_credit', 0))
        preferred_faculty = student_info.get('preferred_faculty')

        # ── 1. Find students in a similar mark band ───────────────────────────
        band_data = None
        for mt, ct in [(10, 20), (20, 40), (100, 200)]:
            mask = (
                (self.data['Final Mark'] >= previous_mark - mt) &
                (self.data['Final Mark'] <= previous_mark + mt) &
                (self.data['Stats Credit'] >= stats_credit - ct) &
                (self.data['Stats Credit'] <= stats_credit + ct)
            )
            band_data = self.data[mask]
            if len(band_data) >= 10:
                break

        passed = band_data[band_data['Pass Fail'] == 'P']

        # ── 2. Faculty recommendations (stable even when passers are few) ─────
        fac_base = band_data.copy()
        fac_base['pass_count'] = (fac_base['Pass Fail'] == 'P').astype(int)
        fac_stats = fac_base.groupby('Faculty Name').agg(
            avg_mark=('Final Mark', 'mean'),
            total=('Final Mark', 'count'),
            student_count=('pass_count', 'sum')
        )
        fac_stats = fac_stats[fac_stats['total'] >= 3]
        fac_stats['pass_rate'] = (fac_stats['student_count'] / fac_stats['total']).round(3)
        fac_stats['avg_mark'] = fac_stats['avg_mark'].round(2)
        fac_stats['profile_fit'] = (1 - (fac_stats['avg_mark'] - previous_mark).abs() / 100).clip(lower=0)
        fac_stats['score'] = (0.7 * fac_stats['pass_rate']) + (0.3 * fac_stats['profile_fit'])
        if preferred_faculty:
            fac_stats['score'] = np.where(fac_stats.index == preferred_faculty, fac_stats['score'] + 0.05, fac_stats['score'])
        fac_stats = fac_stats.sort_values(['score', 'student_count'], ascending=[False, False]).head(n_recommendations)
        fac_stats = fac_stats[['avg_mark', 'student_count', 'pass_rate']].reset_index()
        fac_stats.columns = ['faculty', 'avg_mark', 'student_count', 'pass_rate']

        # ── 3. Department recommendations (stable fallback) ───────────────────
        dept_base = band_data.copy()
        dept_base['pass_count'] = (dept_base['Pass Fail'] == 'P').astype(int)
        dept_stats = dept_base.groupby(['Faculty Name', 'Department']).agg(
            avg_mark=('Final Mark', 'mean'),
            total=('Final Mark', 'count'),
            student_count=('pass_count', 'sum')
        )
        dept_stats = dept_stats[dept_stats['total'] >= 3]
        dept_stats['pass_rate'] = (dept_stats['student_count'] / dept_stats['total']).round(3)
        dept_stats['avg_mark'] = dept_stats['avg_mark'].round(2)
        dept_stats['profile_fit'] = (1 - (dept_stats['avg_mark'] - previous_mark).abs() / 100).clip(lower=0)
        dept_stats['score'] = (0.7 * dept_stats['pass_rate']) + (0.3 * dept_stats['profile_fit'])
        if preferred_faculty:
            dept_stats['score'] = np.where(
                dept_stats.index.get_level_values(0) == preferred_faculty,
                dept_stats['score'] + 0.05,
                dept_stats['score']
            )
        dept_stats = dept_stats.sort_values(['score', 'student_count'], ascending=[False, False]).head(n_recommendations)
        dept_stats = dept_stats[['avg_mark', 'student_count', 'pass_rate']].reset_index()
        dept_stats.columns = ['faculty', 'department', 'avg_mark', 'student_count', 'pass_rate']

        # ── 4. Course recommendations ─────────────────────────────────────────
        top_faculty = fac_stats.iloc[0]['faculty'] if not fac_stats.empty else preferred_faculty
        top_dept    = dept_stats.iloc[0]['department'] if not dept_stats.empty else None
        course_recs = self.recommend_courses(student_info, n_recommendations, top_faculty, top_dept)
        if not course_recs and top_faculty:
            course_recs = self.recommend_courses(student_info, n_recommendations, top_faculty, None)
        if not course_recs:
            course_recs = self.recommend_courses(student_info, n_recommendations, None, None)

        # ── 5. Similar students sample (for display) ──────────────────────────
        similar_display = []
        similar_source = passed if not passed.empty else band_data
        for _, row in similar_source.head(10).iterrows():
            similar_display.append({
                'student_id':       row['Student Number'],
                'similarity_score': round(1 - abs(row['Final Mark'] - previous_mark) / 100, 3),
                'faculty':          row['Faculty Name'],
                'department':       row['Department'],
                'performance':      round(row['Final Mark'], 2),
                'pass_status':      row['Pass Fail']
            })

        return {
            'similar_students':           similar_display,
            'faculty_recommendations':    fac_stats.to_dict('records'),
            'department_recommendations': dept_stats.to_dict('records'),
            'qualification_recommendations': [],
            'course_recommendations':     course_recs,
            'recommendation_version': 2
        }

    def recommend_courses(self, student_info, n_recommendations=5, faculty=None, department=None):
        """Recommend courses using vectorized aggregation for speed."""
        completed_courses = set(student_info.get('completed_courses', set()))

        filtered_data = self.data
        if faculty and department:
            scoped = filtered_data[(filtered_data['Faculty Name'] == faculty) & (filtered_data['Department'] == department)]
            filtered_data = scoped if not scoped.empty else filtered_data[filtered_data['Faculty Name'] == faculty]
        elif faculty:
            scoped = filtered_data[filtered_data['Faculty Name'] == faculty]
            filtered_data = scoped if not scoped.empty else filtered_data
        elif department:
            scoped = filtered_data[filtered_data['Department'] == department]
            filtered_data = scoped if not scoped.empty else filtered_data

        if filtered_data.empty or 'Subject Code' not in filtered_data.columns:
            return []

        grouped = filtered_data.groupby('Subject Code').agg(
            avg_mark=('Final Mark', 'mean'),
            student_count=('Final Mark', 'count'),
            pass_count=('Pass Fail', lambda s: (s == 'P').sum()),
            faculty_name=('Faculty Name', 'first'),
            department_name=('Department', 'first')
        ).reset_index()

        if completed_courses:
            grouped = grouped[~grouped['Subject Code'].isin(completed_courses)]

        if grouped.empty:
            return []

        grouped['pass_rate'] = (grouped['pass_count'] / grouped['student_count']).round(3)
        grouped['avg_mark'] = grouped['avg_mark'].round(2)

        top = grouped.sort_values(['pass_rate', 'avg_mark'], ascending=[False, False]).head(n_recommendations)

        return [
            {
                'course': row['Subject Code'],
                'pass_rate': float(row['pass_rate']),
                'avg_mark': float(row['avg_mark']),
                'student_count': int(row['student_count']),
                'department': row['department_name'],
                'faculty': row['faculty_name']
            }
            for _, row in top.iterrows()
        ]
