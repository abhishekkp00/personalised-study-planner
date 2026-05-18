"""
Real-Time Exam Readiness Predictor
Analyzes student performance and predicts exam readiness
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class ExamReadinessPrediction:
    def __init__(self, exam_date=None):
        """Initialize with exam date (default: 30 days from now)"""
        self.exam_date = exam_date or (datetime.now() + timedelta(days=30))
        self.days_to_exam = (self.exam_date - datetime.now()).days
        
    def calculate_performance_trajectory(self, df):
        """Calculate performance trend using regression"""
        if len(df) < 2:
            return None
        
        # Group by subject and calculate trend
        subjects = df['subject'].unique()
        trajectories = {}
        
        for subject in subjects:
            subject_data = df[df['subject'] == subject].tail(10)  # Last 10 records
            if len(subject_data) < 2:
                continue
            
            X = np.arange(len(subject_data)).reshape(-1, 1)
            y = subject_data['target_score'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]  # Improvement rate per record
            current_avg = y[-1]
            
            trajectories[subject] = {
                'current_score': current_avg,
                'improvement_rate': slope,
                'records_count': len(subject_data)
            }
        
        return trajectories
    
    def predict_exam_performance(self, df, model=None, encoders=None, features=None):
        """Predict exam performance based on current trajectory"""
        trajectories = self.calculate_performance_trajectory(df)
        
        if not trajectories:
            return None
        
        # Project scores to exam date
        predictions = {}
        total_prediction = 0
        
        for subject, data in trajectories.items():
            current = data['current_score']
            rate = data['improvement_rate']
            
            # Estimate days between records (assume weekly reviews)
            estimated_days_between_records = 7
            periods_to_exam = self.days_to_exam / estimated_days_between_records
            
            # Linear projection
            projected_score = current + (rate * periods_to_exam)
            projected_score = np.clip(projected_score, 0, 100)
            
            predictions[subject] = {
                'current': current,
                'projected': projected_score,
                'improvement_potential': projected_score - current,
                'trend': 'improving' if rate > 0.5 else ('stable' if rate > -0.5 else 'declining')
            }
            total_prediction += projected_score
        
        avg_prediction = total_prediction / len(predictions)
        
        return {
            'predictions': predictions,
            'overall_readiness': avg_prediction,
            'days_to_exam': self.days_to_exam,
            'exam_date': self.exam_date.strftime('%Y-%m-%d')
        }
    
    def calculate_readiness_percentage(self, df):
        """Calculate overall readiness as percentage"""
        if len(df) == 0:
            return 0
        
        # Readiness = (average_score / 100) * (1 - decay_factor)
        avg_score = df['target_score'].mean()
        
        # Decay factor: how much time left affects readiness
        decay_factor = max(0, (30 - self.days_to_exam) / 30)  # Full points at day 0, decreases
        
        # Performance factor
        performance_factor = (avg_score / 100) ** 1.2
        
        readiness = (performance_factor * 100) * (1 - decay_factor * 0.3)
        return np.clip(readiness, 0, 100)
    
    def get_subject_readiness(self, df):
        """Get readiness breakdown by subject"""
        if len(df) == 0:
            return {}
        
        subject_readiness = {}
        
        for subject in df['subject'].unique():
            subject_df = df[df['subject'] == subject]
            avg = subject_df['target_score'].mean()
            count = len(subject_df)
            
            subject_readiness[subject] = {
                'score': avg,
                'records': count,
                'readiness': (avg / 100) * 100
            }
        
        return subject_readiness
    
    def get_critical_topics(self, df, threshold=50):
        """Identify topics below threshold that need urgent attention"""
        if len(df) == 0:
            return []
        
        critical = df[df['target_score'] < threshold].groupby('topic').agg({
            'target_score': ['mean', 'count'],
            'difficulty': lambda x: x.mode()[0] if len(x) > 0 else 'Medium'
        }).reset_index()
        
        critical.columns = ['topic', 'avg_score', 'frequency', 'difficulty']
        critical = critical.sort_values('avg_score').head(5)
        
        return critical.to_dict('records') if len(critical) > 0 else []
    
    def get_strengths(self, df, threshold=75):
        """Identify strong topics"""
        if len(df) == 0:
            return []
        
        strong = df[df['target_score'] >= threshold].groupby('topic').agg({
            'target_score': ['mean', 'count']
        }).reset_index()
        
        strong.columns = ['topic', 'avg_score', 'frequency']
        strong = strong.sort_values('avg_score', ascending=False).head(5)
        
        return strong.to_dict('records') if len(strong) > 0 else []
    
    def get_study_recommendations(self, df):
        """Generate personalized study recommendations"""
        critical = self.get_critical_topics(df)
        strengths = self.get_strengths(df)
        trajectories = self.calculate_performance_trajectory(df)
        
        recommendations = {
            'urgent': [],
            'improvement': [],
            'maintain': [],
            'time_allocation': {}
        }
        
        # Urgent: Score < 40
        for topic_data in critical:
            if topic_data['avg_score'] < 40:
                recommendations['urgent'].append({
                    'topic': topic_data['topic'],
                    'current_score': topic_data['avg_score'],
                    'action': f"Focus on fundamentals. Current: {topic_data['avg_score']:.1f}%",
                    'priority': 'CRITICAL'
                })
        
        # Need improvement: 40-70
        for topic_data in critical:
            if 40 <= topic_data['avg_score'] < 70:
                recommendations['improvement'].append({
                    'topic': topic_data['topic'],
                    'current_score': topic_data['avg_score'],
                    'action': f"Practice more problems. Current: {topic_data['avg_score']:.1f}%",
                    'priority': 'HIGH'
                })
        
        # Maintain: > 75
        for topic_data in strengths:
            if topic_data['avg_score'] >= 75:
                recommendations['maintain'].append({
                    'topic': topic_data['topic'],
                    'current_score': topic_data['avg_score'],
                    'action': f"Quick revision before exam. Current: {topic_data['avg_score']:.1f}%",
                    'priority': 'LOW'
                })
        
        # Time allocation
        if trajectories:
            for subject, data in trajectories.items():
                if data['improvement_rate'] < 0:
                    allocation = 40  # More time to struggling subjects
                elif data['current_score'] < 60:
                    allocation = 35
                elif data['current_score'] < 75:
                    allocation = 20
                else:
                    allocation = 5
                
                recommendations['time_allocation'][subject] = allocation
        
        return recommendations
    
    def get_full_readiness_report(self, df):
        """Generate comprehensive readiness report"""
        readiness_pct = self.calculate_readiness_percentage(df)
        predictions = self.predict_exam_performance(df)
        subject_readiness = self.get_subject_readiness(df)
        critical_topics = self.get_critical_topics(df)
        recommendations = self.get_study_recommendations(df)
        
        return {
            'overall_readiness': readiness_pct,
            'days_to_exam': self.days_to_exam,
            'exam_date': self.exam_date.strftime('%Y-%m-%d'),
            'predictions': predictions,
            'subject_readiness': subject_readiness,
            'critical_topics': critical_topics,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
