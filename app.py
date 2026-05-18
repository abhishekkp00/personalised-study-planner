"""Personalized Study Planner - Streamlit Application"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from datetime import datetime, timedelta
from recommendation import RecommendationEngine, PerformanceLevel
from exam_readiness import ExamReadinessPrediction

st.set_page_config(page_title="Study Planner", page_icon="📚", layout="wide")

@st.cache_resource
def load_model_and_encoders():
    base = 'models/'
    path = base + 'random_forest_model.pkl'
    if not os.path.exists(path):
        return None, None, None
    return (
        joblib.load(path),
        joblib.load(base + 'encoders.pkl'),
        joblib.load(base + 'feature_names.pkl')
    )

@st.cache_data
def load_dataset():
    path = 'data/dataset.csv'
    return pd.read_csv(path) if os.path.exists(path) else None

def make_prediction(model, encoders, features, input_data):
    X = pd.DataFrame([input_data], columns=features)
    for col in ['subject', 'topic', 'difficulty', 'confidence_level']:
        if col in X.columns and col in encoders:
            X[col] = encoders[col].transform(X[col])
    return np.clip(model.predict(X)[0], 0, 100)

def page_dashboard():
    st.title("Dashboard")
    
    df = load_dataset()
    model, encoders, features = load_model_and_encoders()
    
    if df is None or model is None:
        st.error("Dataset or model not found.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "Random Forest")
    col2.metric("R² Score", "0.789")
    col3.metric("MAE", "7.02%")
    col4.metric("Train Time", "0.91s")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", len(df))
    col2.metric("Subjects", df['subject'].nunique())
    col3.metric("Topics", df['topic'].nunique())
    col4.metric("Features", len(features))
    
    st.markdown("---")
    st.subheader("Feature Importance")
    
    importance = pd.DataFrame({
        'Feature': ['previous_score', 'confidence_level', 'difficulty', 'days_since_revision', 
                   'quiz_score', 'study_hours', 'attempts', 'attendance', 'topic', 'subject'],
        'Importance': [0.3444, 0.2022, 0.1002, 0.0968, 0.0926, 0.0867, 0.0332, 0.0263, 0.0110, 0.0066]
    })
    
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df['subject'].value_counts(), title="Samples by Subject")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        order = ['Easy', 'Medium', 'Hard']
        counts = df['difficulty'].value_counts().reindex(order)
        fig = px.bar(counts, title="Samples by Difficulty", 
                    color=order, color_discrete_map={'Easy': '#6bcf7f', 'Medium': '#ffd93d', 'Hard': '#ff6b6b'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min Score", f"{df['target_score'].min():.1f}%")
    col2.metric("Max Score", f"{df['target_score'].max():.1f}%")
    col3.metric("Mean Score", f"{df['target_score'].mean():.1f}%")
    col4.metric("Std Dev", f"{df['target_score'].std():.1f}%")
    
    fig = px.histogram(df, x='target_score', nbins=30, title="Score Distribution")
    st.plotly_chart(fig, use_container_width=True)

def page_predict():
    st.title("Predict Performance")
    
    model, encoders, features = load_model_and_encoders()
    if model is None:
        st.error("Model not found.")
        return
    
    df = load_dataset()
    subjects = sorted(df['subject'].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        subject = st.selectbox("Subject", subjects)
        topics = sorted(df[df['subject'] == subject]['topic'].unique())
        topic = st.selectbox("Topic", topics)
    with col2:
        prev_score = st.slider("Previous Score (%)", 0, 100, 60)
        study_hrs = st.slider("Study Hours/Week", 0.0, 15.0, 5.0)
    
    col1, col2 = st.columns(2)
    with col1:
        attempts = st.number_input("Practice Attempts", 1, 50, 5)
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    with col2:
        confidence = st.selectbox("Confidence Level", ["Low", "Medium", "High"])
        days_rev = st.slider("Days Since Revision", 0, 90, 15)
    
    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
    with col2:
        quiz_score = st.slider("Quiz Score (%)", 0, 100, 65)
    
    if st.button("Predict Performance"):
        data = {
            'student_id': 0,
            'subject': subject, 'topic': topic,
            'previous_score': prev_score,
            'study_hours': study_hrs,
            'attempts': attempts,
            'difficulty': difficulty,
            'confidence_level': confidence,
            'days_since_revision': days_rev,
            'attendance': attendance,
            'quiz_score': quiz_score
        }
        
        with st.spinner("Predicting..."):
            score = make_prediction(model, encoders, features, data)
        
        engine = RecommendationEngine()
        perf = engine.classify_performance(score)
        priority = engine.calculate_priority_score(score, difficulty, days_rev)
        hours = engine.calculate_recommended_hours(perf, difficulty)
        
        st.success("Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Score", f"{score:.1f}%")
        col2.metric("Level", perf.value)
        col3.metric("Priority", f"{priority:.1f}")
        
        st.subheader("Recommendations")
        
        student_data = {
            'student_id': 'USER001',
            'subject': subject,
            'topic': topic,
            'difficulty': difficulty,
            'days_since_revision': days_rev,
            'quiz_score': quiz_score
        }
        
        rec = engine.generate_full_recommendation(student_data, score)
        
        st.info(f"Recommended Study Time: {hours['recommended_hours']} hours/week")
        
        for r in rec['recommendations']:
            if r.startswith('⚠️'):
                st.warning(r)
            elif r.startswith('✅'):
                st.success(r)
            elif r.startswith('•'):
                st.write(r)
            else:
                st.write(r)
        
        st.subheader("Weekly Plan")
        for day in rec['weekly_plan']:
            st.write(f"• {day}")

def page_study_plan():
    st.title("Study Plan")
    
    df = load_dataset()
    model, encoders, features = load_model_and_encoders()
    
    if df is None or model is None:
        st.error("Data or model not found.")
        return
    
    subjects = sorted(df['subject'].unique())
    subject = st.selectbox("Select Subject", subjects)
    
    st.subheader(f"Study Plan for {subject}")
    
    topics_df = df[df['subject'] == subject].drop_duplicates('topic').head(5)
    
    with st.spinner("Generating plan..."):
        engine = RecommendationEngine()
        
        for _, row in topics_df.iterrows():
            data = {
                'student_id': 0,
                'subject': row['subject'],
                'topic': row['topic'],
                'previous_score': row['previous_score'],
                'study_hours': row['study_hours'],
                'attempts': row['attempts'],
                'difficulty': row['difficulty'],
                'confidence_level': row['confidence_level'],
                'days_since_revision': row['days_since_revision'],
                'attendance': row['attendance'],
                'quiz_score': row['quiz_score']
            }
            
            score = make_prediction(model, encoders, features, data)
            
            student_data = {
                'student_id': 'USER',
                'subject': row['subject'],
                'topic': row['topic'],
                'difficulty': row['difficulty'],
                'days_since_revision': row['days_since_revision'],
                'quiz_score': row['quiz_score']
            }
            
            rec = engine.generate_full_recommendation(student_data, score)
            
            st.markdown(f"### {rec['topic']}")
            st.write(f"**Score:** {rec['predicted_score']}% | **Level:** {rec['performance_level']} | **Priority:** {rec['priority_score']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Study Time: {rec['recommended_hours']} hrs/week")
                st.write(f"Difficulty: {rec['difficulty']}")
            with col2:
                st.write("Weekly Schedule:")
                for day in rec['weekly_plan'][:3]:
                    st.caption(day)
            
            st.markdown("---")

def page_analytics():
    st.title("Analytics")
    
    df = load_dataset()
    if df is None:
        st.error("Data not found.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Distribution", "Subjects", "Features"])
    
    with tab1:
        st.subheader("Student Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            data = pd.DataFrame({
                'Week': list(range(1, 9)),
                'Quiz': [45, 52, 58, 62, 68, 72, 75, 78],
                'Practice': [40, 48, 55, 60, 65, 70, 74, 77]
            })
            fig = px.line(data, x='Week', y=['Quiz', 'Practice'], markers=True, title="8-Week Progress")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            subjects = pd.DataFrame({
                'Subject': ['Math', 'Science', 'English', 'History', 'Chemistry', 'Physics', 'Biology'],
                'Score': [72, 68, 75, 60, 65, 70, 73]
            })
            fig = px.bar(subjects, x='Subject', y='Score', title="Performance by Subject", range_y=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Score Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='target_score', nbins=30, title='Score Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            order = ['Easy', 'Medium', 'Hard']
            df_sorted = df.copy()
            df_sorted['difficulty'] = pd.Categorical(df_sorted['difficulty'], categories=order, ordered=True)
            fig = px.box(df_sorted.sort_values('difficulty'), x='difficulty', y='target_score', title='Scores by Difficulty')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mean", f"{df['target_score'].mean():.1f}%")
        col2.metric("Median", f"{df['target_score'].median():.1f}%")
        col3.metric("Std Dev", f"{df['target_score'].std():.1f}%")
        col4.metric("Min", f"{df['target_score'].min():.1f}%")
        col5.metric("Max", f"{df['target_score'].max():.1f}%")
    
    with tab3:
        st.subheader("Subject Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            subject_stats = df.groupby('subject')['target_score'].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=False)
            fig = px.bar(subject_stats, x='subject', y='mean', title='Avg Score by Subject', range_y=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(subject_stats, values='count', names='subject', title='Sample Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top 15 Topics")
        topic_stats = df.groupby('topic')['target_score'].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=False).head(15)
        fig = px.bar(topic_stats, x='mean', y='topic', orientation='h', title='Top Topics by Score', range_x=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Importance")
        
        col1, col2 = st.columns(2)
        with col1:
            importance = pd.DataFrame({
                'Feature': ['Previous Score', 'Confidence', 'Difficulty', 'Days Revision', 'Quiz Score', 'Study Hours', 'Attempts', 'Attendance'],
                'Importance': [0.3444, 0.2022, 0.1002, 0.0968, 0.0926, 0.0867, 0.0332, 0.0263]
            })
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            cols = ['previous_score', 'study_hours', 'attempts', 'days_since_revision', 'attendance', 'quiz_score']
            corr = df[cols + ['target_score']].corr()['target_score'].drop('target_score').sort_values(ascending=False)
            fig = px.bar(x=corr.values, y=corr.index, orientation='h', title='Feature Correlation', range_x=[-1, 1])
            st.plotly_chart(fig, use_container_width=True)

def page_exam_readiness():
    st.title("Exam Readiness Predictor")
    
    df = load_dataset()
    if df is None:
        st.error("Data not found.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        exam_days = st.number_input("Days until exam", 1, 365, 30)
    with col2:
        st.write("")
    with col3:
        st.write("")
    
    exam_date = datetime.now() + timedelta(days=exam_days)
    predictor = ExamReadinessPrediction(exam_date=exam_date)
    report = predictor.get_full_readiness_report(df)
    
    # Overall readiness
    readiness = report['overall_readiness']
    st.markdown(f"## Overall Exam Readiness: {readiness:.1f}%")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Days to Exam", report['days_to_exam'])
    col2.metric("Exam Date", report['exam_date'])
    col3.metric("Subjects", len(report['subject_readiness']))
    col4.metric("Status", "Ready" if readiness >= 75 else ("On Track" if readiness >= 50 else "Needs Work"))
    
    st.markdown("---")
    
    # Readiness gauge
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Readiness by Subject")
        subject_data = report['subject_readiness']
        if subject_data:
            subjects = list(subject_data.keys())
            scores = [subject_data[s]['score'] for s in subjects]
            fig = px.bar(x=subjects, y=scores, title="Current Performance", range_y=[0, 100],
                        color=scores, color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Score Trajectory")
        if report['predictions'] and 'predictions' in report['predictions']:
            preds = report['predictions']['predictions']
            subjects = list(preds.keys())
            current = [preds[s]['current'] for s in subjects]
            projected = [preds[s]['projected'] for s in subjects]
            
            trajectory_df = pd.DataFrame({
                'Subject': subjects + subjects,
                'Score': current + projected,
                'Type': ['Current']*len(subjects) + ['Projected']*len(subjects)
            })
            fig = px.bar(trajectory_df, x='Subject', y='Score', color='Type', barmode='group', range_y=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Critical topics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Critical Topics (Urgent)")
        critical = report['critical_topics']
        if critical:
            for topic in critical[:5]:
                with st.container():
                    st.write(f"**{topic['topic']}**")
                    st.write(f"Current: {topic['avg_score']:.1f}% | Difficulty: {topic['difficulty']}")
                    if topic['avg_score'] < 40:
                        st.error("CRITICAL - Focus here first!")
                    elif topic['avg_score'] < 60:
                        st.warning("Needs improvement")
        else:
            st.success("No critical topics!")
    
    with col2:
        st.subheader("Recommendations")
        recs = report['recommendations']
        
        if recs['urgent']:
            st.subheader("🚨 Urgent Action")
            for rec in recs['urgent'][:3]:
                st.error(f"{rec['topic']}: {rec['action']}")
        
        if recs['improvement']:
            st.subheader("⚠️ Need Improvement")
            for rec in recs['improvement'][:3]:
                st.warning(f"{rec['topic']}: {rec['action']}")
    
    st.markdown("---")
    
    # Study time allocation
    st.subheader("Recommended Study Time Allocation")
    if report['recommendations']['time_allocation']:
        allocation = report['recommendations']['time_allocation']
        time_df = pd.DataFrame({
            'Subject': list(allocation.keys()),
            'Allocation %': list(allocation.values())
        })
        fig = px.pie(time_df, values='Allocation %', names='Subject', title='Weekly Study Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Prediction accuracy
    st.subheader("Performance Prediction")
    if report['predictions']:
        pred = report['predictions']
        if 'overall_readiness' in pred:
            col1, col2 = st.columns(2)
            col1.metric("Predicted Exam Score", f"{pred['overall_readiness']:.1f}%")
            col2.metric("Confidence", "High" if report['overall_readiness'] >= 70 else "Medium")

def main():
    st.sidebar.title("Study Planner")
    
    model_path = 'models/random_forest_model.pkl'
    if not os.path.exists(model_path):
        st.sidebar.warning("Model not trained yet.")
    
    page = st.sidebar.radio("Navigate to:", 
        ["Dashboard", "Predict", "Study Plan", "Exam Readiness", "Analytics"])
    
    if page == "Dashboard":
        page_dashboard()
    elif page == "Predict":
        page_predict()
    elif page == "Study Plan":
        page_study_plan()
    elif page == "Exam Readiness":
        page_exam_readiness()
    else:
        page_analytics()

if __name__ == "__main__":
    main()
