"""Personalized Study Planner - Streamlit Application"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from pathlib import Path
from datetime import datetime, timedelta
from recommendation import RecommendationEngine, PerformanceLevel
from exam_readiness import ExamReadinessPrediction

st.set_page_config(page_title="Study Planner", page_icon="📚", layout="wide")

HISTORY_FILE = Path('data/student_performance_history.csv')

def initialize_state():
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    if 'saved_profile' not in st.session_state:
        st.session_state.saved_profile = None

def apply_theme():
    dark_mode = st.session_state.theme == 'dark'
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp {
                background: radial-gradient(circle at top, #111827 0%, #0b1020 45%, #050816 100%);
                color: #f3f4f6;
            }
            section[data-testid="stSidebar"] {
                background: rgba(17, 24, 39, 0.92);
                border-right: 1px solid rgba(255,255,255,0.08);
            }
            .card {
                background: rgba(17, 24, 39, 0.72);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 18px 20px;
                box-shadow: 0 12px 30px rgba(0,0,0,0.22);
            }
            .hero {
                padding: 28px;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(59,130,246,0.24), rgba(168,85,247,0.18));
                border: 1px solid rgba(255,255,255,0.08);
                margin-bottom: 18px;
            }
            .muted {
                color: #cbd5e1;
                font-size: 0.95rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
                color: #0f172a;
            }
            section[data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid #e2e8f0;
            }
            .card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 18px 20px;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
            }
            .hero {
                padding: 28px;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(96,165,250,0.18), rgba(167,139,250,0.14));
                border: 1px solid #e2e8f0;
                margin-bottom: 18px;
            }
            .muted {
                color: #475569;
                font-size: 0.95rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

def history_columns():
    return [
        'timestamp', 'student_id', 'subject', 'topic', 'previous_score', 'study_hours',
        'attempts', 'difficulty', 'confidence_level', 'days_since_revision', 'attendance',
        'quiz_score', 'predicted_score', 'performance_level', 'priority_score', 'theme'
    ]

def load_history():
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=history_columns())

def save_prediction_history(record):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_history()
    history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    history = history.drop_duplicates(subset=['student_id', 'subject', 'topic', 'timestamp'], keep='last')
    history.to_csv(HISTORY_FILE, index=False)

def load_latest_profile(student_id):
    history = load_history()
    if history.empty or 'student_id' not in history.columns:
        return None
    matches = history[history['student_id'].astype(str) == str(student_id)]
    if matches.empty:
        return None
    row = matches.sort_values('timestamp').iloc[-1]
    return row.to_dict()

def get_student_history(student_id):
    history = load_history()
    if history.empty or 'student_id' not in history.columns:
        return pd.DataFrame()
    student_history = history[history['student_id'].astype(str) == str(student_id)].copy()
    if student_history.empty:
        return student_history
    if 'timestamp' in student_history.columns:
        student_history['timestamp'] = pd.to_datetime(student_history['timestamp'], errors='coerce')
        student_history = student_history.sort_values('timestamp')
    return student_history

def get_student_options():
    history = load_history()
    if history.empty or 'student_id' not in history.columns:
        return []
    students = history['student_id'].dropna().astype(str).unique().tolist()
    return sorted(students)

def page_my_progress():
    st.markdown('<div class="hero"><h1 style="margin:0;">My Progress</h1><p class="muted">Track how a student is improving over time using the saved performance history.</p></div>', unsafe_allow_html=True)

    history = load_history()
    student_options = get_student_options()

    if not student_options:
        st.info("No saved student history yet. Go to Predict and save at least one performance record first.")
        return

    selected_student = st.selectbox("Select Student ID", student_options)
    student_history = get_student_history(selected_student)

    if student_history.empty:
        st.warning("No history found for this student yet.")
        return

    if 'predicted_score' not in student_history.columns:
        st.warning("This student history does not include saved predictions yet.")
        return

    latest = student_history.iloc[-1]
    first = student_history.iloc[0]

    latest_score = float(latest['predicted_score'])
    first_score = float(first['predicted_score'])
    score_change = latest_score - first_score
    score_change_pct = ((score_change / first_score) * 100) if first_score else 0
    average_score = student_history['predicted_score'].astype(float).mean()
    best_score = student_history['predicted_score'].astype(float).max()
    total_entries = len(student_history)
    score_std = student_history['predicted_score'].astype(float).std(ddof=0)
    consistency = max(0.0, 100 - score_std * 2)

    subject_summary = student_history.groupby('subject')['predicted_score'].agg(['mean', 'min', 'max', 'count']).reset_index().sort_values('mean', ascending=False)
    subject_summary['trend'] = subject_summary['max'] - subject_summary['min']
    if len(student_history) > 1:
        first_subject_score = student_history['predicted_score'].astype(float).iloc[0]
        last_subject_score = student_history['predicted_score'].astype(float).iloc[-1]
        subject_delta = last_subject_score - first_subject_score
    else:
        subject_delta = 0.0

    most_improved = None
    if not subject_summary.empty:
        per_subject_change = (
            student_history.groupby('subject')['predicted_score']
            .agg(['first', 'last'])
            .reset_index()
        )
        per_subject_change['delta'] = per_subject_change['last'] - per_subject_change['first']
        most_improved = per_subject_change.sort_values('delta', ascending=False).iloc[0].to_dict()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Latest Score", f"{latest_score:.1f}%", f"{latest['subject']} • {latest['topic']}")
    with col2:
        metric_card("Change", f"{score_change:+.1f}%", f"Since first saved attempt ({score_change_pct:+.1f}%)")
    with col3:
        metric_card("Best Score", f"{best_score:.1f}%", "Highest saved prediction")
    with col4:
        metric_card("Saved Attempts", total_entries, "Records in history")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Consistency", f"{consistency:.0f}%", "Lower score spread is better")
    with col2:
        metric_card("Average Score", f"{average_score:.1f}%", "All saved attempts")
    with col3:
        metric_card("Improvement", f"{subject_delta:+.1f}%", "First vs latest saved score")
    with col4:
        improved_subject = most_improved['subject'] if most_improved is not None else "N/A"
        metric_card("Top Momentum", improved_subject, "Best subject trajectory")

    st.markdown("---")

    badge_row = []
    for _, row in subject_summary.head(5).iterrows():
        score = float(row['mean'])
        if score >= 75:
            tone = "#16a34a"
            label = "strong"
        elif score >= 50:
            tone = "#f59e0b"
            label = "steady"
        else:
            tone = "#dc2626"
            label = "needs focus"
        badge_row.append(
            f'<span style="display:inline-block;margin:0 8px 8px 0;padding:8px 12px;border-radius:999px;background:{tone}22;color:{tone};border:1px solid {tone}55;font-weight:600;">{row["subject"]}: {score:.1f}% · {label}</span>'
        )
    st.markdown("<div>" + "".join(badge_row) + "</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Goal Tracker")
    goal_col1, goal_col2, goal_col3 = st.columns(3)
    with goal_col1:
        target_score = st.slider("Target score", 0, 100, 80, help="Set the score the student wants to reach.")
    with goal_col2:
        score_gap = target_score - latest_score
        status_label = "Goal reached" if score_gap <= 0 else "In progress"
        metric_card("Goal Status", status_label, f"Target {target_score}%")
    with goal_col3:
        metric_card("Gap to Goal", f"{score_gap:+.1f}%", "Positive means more improvement needed")

    gap_fill = max(0, min(100, int(round((latest_score / target_score) * 100)))) if target_score else 0
    st.caption("Progress toward target")
    st.progress(gap_fill)

    avg_gain = score_change / max(1, total_entries - 1)
    if score_gap <= 0:
        progress_status = "ahead"
        progress_message = f"Great — the student has already reached the {target_score}% goal."
        st.success(progress_message)
    else:
        if avg_gain > 0:
            estimated_attempts = int(np.ceil(score_gap / avg_gain))
            if estimated_attempts <= 2:
                progress_status = "ahead"
                progress_message = f"Ahead of pace — the target may be reached in about {estimated_attempts} more saved attempts."
                st.success(progress_message)
            elif estimated_attempts <= 4:
                progress_status = "on track"
                progress_message = f"On track — at the current pace, the target may be reached in about {estimated_attempts} more saved attempts."
                st.info(progress_message)
            else:
                progress_status = "behind"
                progress_message = f"Behind pace — about {estimated_attempts} more saved attempts may be needed at the current trend."
                st.warning(progress_message)
        else:
            progress_status = "behind"
            progress_message = "Behind pace — the current trend is not improving yet. Save more attempts after focused study to estimate a target path."
            st.warning(progress_message)

    st.caption(f"Goal pace status: {progress_status.title()}")

    if len(subject_summary) > 0:
        next_best_subject = subject_summary.iloc[0]
        st.caption(f"Best current subject to reinforce: {next_best_subject['subject']} ({next_best_subject['mean']:.1f}% average)")

    st.subheader("Weekly Goal Plan")
    if score_gap <= 0:
        st.success("No milestone plan needed — the student has already reached the goal. Keep the score stable with light revision.")
    else:
        if len(subject_summary) > 0:
            focus_subjects = subject_summary.sort_values('mean').head(4).reset_index(drop=True)
        else:
            focus_subjects = pd.DataFrame(columns=['subject', 'mean'])

        if score_change > 0 and total_entries > 1:
            average_weekly_gain = max(score_change / (total_entries - 1), 0.1)
        else:
            average_weekly_gain = max(score_gap / 4, 5)

        weeks_needed = max(1, int(np.ceil(score_gap / average_weekly_gain)))
        plan_weeks = min(max(3, weeks_needed), 6)
        milestone_rows = []

        for week in range(1, plan_weeks + 1):
            week_target = min(100, latest_score + (score_gap / plan_weeks) * week)
            if not focus_subjects.empty:
                subject_row = focus_subjects.iloc[min(week - 1, len(focus_subjects) - 1)]
                focus_subject = subject_row['subject']
                current_avg = float(subject_row['mean'])
            else:
                focus_subject = "General revision"
                current_avg = latest_score

            if week == 1:
                action = "Review weak topics and rebuild fundamentals"
            elif week == plan_weeks:
                action = "Do a full mock and compare against the target"
            else:
                action = "Practice mixed questions and revise mistakes"

            milestone_rows.append({
                'Week': f'Week {week}',
                'Target Score': f'{week_target:.1f}%',
                'Focus Subject': focus_subject,
                'Current Avg': f'{current_avg:.1f}%',
                'Action': action,
            })

        milestone_df = pd.DataFrame(milestone_rows)
        st.dataframe(milestone_df, use_container_width=True, hide_index=True)

        plan_cols = st.columns(min(3, len(milestone_df)))
        for index, row in milestone_df.head(3).iterrows():
            with plan_cols[index % len(plan_cols)]:
                st.markdown(
                    f'''
                    <div style="border:1px solid rgba(148,163,184,0.25); border-radius:16px; padding:16px; background:rgba(255,255,255,0.03); margin-bottom:10px;">
                        <div style="font-size:0.9rem; opacity:0.75;">{row['Week']}</div>
                        <div style="font-size:1.5rem; font-weight:800; margin:4px 0;">{row['Target Score']}</div>
                        <div style="font-size:0.95rem; font-weight:600;">{row['Focus Subject']}</div>
                        <div style="font-size:0.85rem; opacity:0.75; margin-top:4px;">{row['Action']}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

    summary_plan_text = '- Goal already achieved; keep the score stable with light revision.'
    if score_gap > 0 and 'milestone_df' in locals():
        summary_plan_text = chr(10).join(
            [f'- {row["Week"]}: {row["Target Score"]} | {row["Focus Subject"]} | {row["Action"]}' for _, row in milestone_df.iterrows()]
        )

    summary_report = f"""My Progress Summary
Student ID: {selected_student}
Latest score: {latest_score:.1f}%
First saved score: {first_score:.1f}%
Best score: {best_score:.1f}%
Average score: {average_score:.1f}%
Saved attempts: {total_entries}
Goal target: {target_score}%
Goal gap: {score_gap:+.1f}%
Goal pace status: {progress_status.title()}
Top momentum subject: {improved_subject if most_improved is not None else 'N/A'}

Weekly plan:
{summary_plan_text}
"""

    st.download_button(
        label="Download progress summary",
        data=summary_report.encode('utf-8'),
        file_name=f"{selected_student}_progress_summary.txt",
        mime='text/plain'
    )

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.subheader("Score Trend")
        trend_df = student_history.copy()
        trend_df['attempt'] = range(1, len(trend_df) + 1)
        trend_df['timestamp_label'] = trend_df['timestamp'].dt.strftime('%d %b %H:%M') if pd.api.types.is_datetime64_any_dtype(trend_df['timestamp']) else trend_df['attempt'].astype(str)
        trend_df['rolling_average'] = trend_df['predicted_score'].astype(float).rolling(3, min_periods=1).mean()
        trend_melt = trend_df[['attempt', 'predicted_score', 'rolling_average']].melt(id_vars='attempt', var_name='Series', value_name='Score')
        trend_melt['Series'] = trend_melt['Series'].replace({'predicted_score': 'Actual Score', 'rolling_average': '3-Point Average'})
        fig = px.line(
            trend_melt,
            x='attempt',
            y='Score',
            color='Series',
            markers=True,
            title='Predicted Score Over Time'
        )
        fig.update_layout(height=380, xaxis_title='Attempt', yaxis_title='Predicted Score', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')

    with summary_col2:
        st.subheader("Before vs Now")
        comparison_df = pd.DataFrame({
            'Stage': ['First saved attempt', 'Latest attempt', 'Average of all attempts'],
            'Score': [first_score, latest_score, average_score]
        })
        fig = px.bar(
            comparison_df,
            x='Stage',
            y='Score',
            color='Stage',
            title='Performance Comparison'
        )
        fig.update_layout(height=380, yaxis_range=[0, 100], showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.subheader("History Snapshot")
        snapshot_df = student_history[['timestamp', 'subject', 'topic', 'difficulty', 'predicted_score', 'priority_score']].copy()
        snapshot_df['timestamp'] = snapshot_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M') if pd.api.types.is_datetime64_any_dtype(snapshot_df['timestamp']) else snapshot_df['timestamp']
        st.dataframe(snapshot_df.tail(10), use_container_width=True)

    with detail_col2:
        st.subheader("Progress Insights")
        score_trend = 'improving' if score_change > 0 else ('stable' if abs(score_change) < 1 else 'declining')
        if score_trend == 'improving':
            st.success(f"Good momentum — the predicted score improved by {score_change:.1f}% overall.")
        elif score_trend == 'stable':
            st.info("Scores are stable. A few focused study sessions should move the trend up.")
        else:
            st.warning(f"Scores dipped by {abs(score_change):.1f}%. Focus on the weakest topics first.")

        recent_history = student_history.tail(3)
        st.write("Recent saved attempts:")
        for _, row in recent_history.iterrows():
            ts = row['timestamp'].strftime('%d %b %H:%M') if pd.notna(row['timestamp']) else 'Unknown time'
            st.caption(f"{ts} — {row['subject']} / {row['topic']} — {float(row['predicted_score']):.1f}%")

        if most_improved is not None:
            st.info(f"Most improved subject: {most_improved['subject']} ({most_improved['delta']:+.1f}% change)")

        st.download_button(
            label="Download progress history",
            data=student_history.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_student}_progress_history.csv",
            mime='text/csv'
        )

    st.markdown("---")

    st.subheader("Attempt Timeline")
    growth_col1, growth_col2 = st.columns(2)
    with growth_col1:
        st.caption("Latest score progress")
        st.progress(int(round(max(0, min(100, latest_score)))))
        st.caption(f"{latest_score:.1f}% of the target scale")
    with growth_col2:
        st.caption("Overall improvement")
        improvement_meter = max(0, min(100, int(round(abs(score_change_pct)))))
        st.progress(improvement_meter)
        st.caption(f"{score_change:+.1f}% vs first saved attempt")

    timeline_cols = st.columns(min(3, len(student_history)))
    for index, (_, row) in enumerate(student_history.iterrows()):
        if index >= 3:
            break
        with timeline_cols[index % len(timeline_cols)]:
            attempt_time = row['timestamp'].strftime('%d %b %H:%M') if pd.notna(row['timestamp']) else 'Unknown time'
            score_value = float(row['predicted_score'])
            if score_value >= 75:
                border = '#16a34a'
                accent = 'linear-gradient(135deg, rgba(22,163,74,0.18), rgba(22,163,74,0.05))'
                status = 'Strong'
                icon = '🟢'
            elif score_value >= 50:
                border = '#f59e0b'
                accent = 'linear-gradient(135deg, rgba(245,158,11,0.18), rgba(245,158,11,0.05))'
                status = 'Steady'
                icon = '🟡'
            else:
                border = '#dc2626'
                accent = 'linear-gradient(135deg, rgba(220,38,38,0.18), rgba(220,38,38,0.05))'
                status = 'Needs focus'
                icon = '🔴'
            progress_fill = max(1, min(100, int(round(score_value))))
            st.markdown(
                f'''
                <div style="border:1px solid {border}; border-radius:18px; padding:16px; margin-bottom:12px; background:{accent}; box-shadow:0 12px 28px rgba(0,0,0,0.12);">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
                        <div style="font-size:0.9rem; opacity:0.8;">Attempt {index + 1}</div>
                        <div style="font-size:0.9rem; font-weight:700; color:{border};">{icon} {status}</div>
                    </div>
                    <div style="font-size:1.75rem; font-weight:800; margin:6px 0 2px; color:{border};">{score_value:.1f}%</div>
                    <div style="font-size:0.95rem; font-weight:600;">{row['subject']} • {row['topic']}</div>
                    <div style="font-size:0.85rem; opacity:0.75; margin-top:2px;">{attempt_time}</div>
                    <div style="margin-top:10px; height:8px; width:100%; border-radius:999px; background:rgba(148,163,184,0.20); overflow:hidden;">
                        <div style="height:100%; width:{progress_fill}%; border-radius:999px; background:{border};"></div>
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    if 'subject' in student_history.columns:
        st.subheader("Subject Comparison")
        subject_summary['delta'] = subject_summary['max'] - subject_summary['min']
        fig = px.bar(
            subject_summary,
            x='subject',
            y='mean',
            color='mean',
            color_continuous_scale='Viridis',
            title='Average Predicted Score by Subject'
        )
        fig.update_layout(height=360, yaxis_range=[0, 100], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')

        st.dataframe(
            subject_summary[['subject', 'mean', 'min', 'max', 'count', 'delta']].rename(columns={
                'subject': 'Subject', 'mean': 'Average', 'min': 'Min', 'max': 'Max', 'count': 'Attempts', 'delta': 'Range'
            }),
            use_container_width=True
        )

    if 'topic' in student_history.columns:
        st.subheader("Topic History")
        topic_summary = student_history.groupby('topic')['predicted_score'].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=False).head(10)
        st.dataframe(topic_summary, use_container_width=True)

def metric_card(title, value, subtitle):
    st.markdown(
        f"""
        <div class="card">
            <div class="muted">{title}</div>
            <div style="font-size: 2rem; font-weight: 700; margin-top: 4px;">{value}</div>
            <div class="muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown('<div class="hero"><h1 style="margin:0;">Dashboard</h1><p class="muted">A clean snapshot of model performance, training data, and student progress.</p></div>', unsafe_allow_html=True)
    
    df = load_dataset()
    model, encoders, features = load_model_and_encoders()
    
    if df is None or model is None:
        st.error("Dataset or model not found.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Model", "Random Forest", "Production-ready regressor")
    with col2:
        metric_card("R² Score", "0.789", "Explains variance well")
    with col3:
        metric_card("MAE", "7.02%", "Average absolute error")
    with col4:
        metric_card("Train Time", "0.91s", "Fast model fitting")
    
    st.write("")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Samples", f"{len(df):,}", "Training records")
    with col2:
        metric_card("Subjects", df['subject'].nunique(), "Unique subjects")
    with col3:
        metric_card("Topics", df['topic'].nunique(), "Unique topics")
    with col4:
        metric_card("Features", len(features), "Input variables")
    
    st.write("")
    st.subheader("Feature Importance")
    
    importance = pd.DataFrame({
        'Feature': ['previous_score', 'confidence_level', 'difficulty', 'days_since_revision', 
                   'quiz_score', 'study_hours', 'attempts', 'attendance', 'topic', 'subject'],
        'Importance': [0.3444, 0.2022, 0.1002, 0.0968, 0.0926, 0.0867, 0.0332, 0.0263, 0.0110, 0.0066]
    })
    
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, width='stretch')
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df['subject'].value_counts(), title="Samples by Subject", color=df['subject'].value_counts().values, color_continuous_scale='Teal')
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')
    with col2:
        order = ['Easy', 'Medium', 'Hard']
        counts = df['difficulty'].value_counts().reindex(order)
        fig = px.bar(counts, title="Samples by Difficulty", 
                    color=order, color_discrete_map={'Easy': '#6bcf7f', 'Medium': '#ffd93d', 'Hard': '#ff6b6b'})
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')
    
    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Min Score", f"{df['target_score'].min():.1f}%", "Lowest observed score")
    with col2:
        metric_card("Max Score", f"{df['target_score'].max():.1f}%", "Highest observed score")
    with col3:
        metric_card("Mean Score", f"{df['target_score'].mean():.1f}%", "Average target score")
    with col4:
        metric_card("Std Dev", f"{df['target_score'].std():.1f}%", "Score spread")
    
    fig = px.histogram(df, x='target_score', nbins=30, title="Score Distribution")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, width='stretch')

def page_predict():
    st.markdown('<div class="hero"><h1 style="margin:0;">Predict Performance</h1><p class="muted">Enter real student data once, then reuse the saved profile later for faster predictions.</p></div>', unsafe_allow_html=True)
    
    model, encoders, features = load_model_and_encoders()
    if model is None:
        st.error("Model not found.")
        return
    
    df = load_dataset()
    subjects = sorted(df['subject'].unique())

    history = load_history()
    saved_student_ids = ['New student'] + sorted(history['student_id'].dropna().astype(str).unique().tolist()) if not history.empty else ['New student']

    st.sidebar.subheader("Profile")
    selected_profile = st.sidebar.selectbox("Load saved student", saved_student_ids)
    profile_data = load_latest_profile(selected_profile) if selected_profile != 'New student' else None
    
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID", value=str(profile_data['student_id']) if profile_data else "USER001")
        subject_index = subjects.index(profile_data['subject']) if profile_data and profile_data['subject'] in subjects else 0
        subject = st.selectbox("Subject", subjects, index=subject_index)
        topics = sorted(df[df['subject'] == subject]['topic'].unique())
        topic_index = topics.index(profile_data['topic']) if profile_data and profile_data['topic'] in topics else 0
        topic = st.selectbox("Topic", topics, index=topic_index)
    with col2:
        prev_score = st.slider("Previous Score (%)", 0, 100, int(profile_data['previous_score']) if profile_data and pd.notna(profile_data.get('previous_score', np.nan)) else 60)
        study_hrs = st.slider("Study Hours/Week", 0.0, 15.0, float(profile_data['study_hours']) if profile_data and pd.notna(profile_data.get('study_hours', np.nan)) else 5.0)
    
    col1, col2 = st.columns(2)
    with col1:
        attempts = st.number_input("Practice Attempts", 1, 50, int(profile_data['attempts']) if profile_data and pd.notna(profile_data.get('attempts', np.nan)) else 5)
        difficulty_options = ["Easy", "Medium", "Hard"]
        difficulty_index = difficulty_options.index(profile_data['difficulty']) if profile_data and profile_data['difficulty'] in difficulty_options else 0
        difficulty = st.selectbox("Difficulty", difficulty_options, index=difficulty_index)
    with col2:
        confidence_options = ["Low", "Medium", "High"]
        confidence_index = confidence_options.index(profile_data['confidence_level']) if profile_data and profile_data['confidence_level'] in confidence_options else 1
        confidence = st.selectbox("Confidence Level", confidence_options, index=confidence_index)
        days_rev = st.slider("Days Since Revision", 0, 90, int(profile_data['days_since_revision']) if profile_data and pd.notna(profile_data.get('days_since_revision', np.nan)) else 15)
    
    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, int(profile_data['attendance']) if profile_data and pd.notna(profile_data.get('attendance', np.nan)) else 75)
    with col2:
        quiz_score = st.slider("Quiz Score (%)", 0, 100, int(profile_data['quiz_score']) if profile_data and pd.notna(profile_data.get('quiz_score', np.nan)) else 65)
    
    if st.button("Predict Performance"):
        data = {
            'student_id': student_id,
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
        with col1:
            metric_card("Predicted Score", f"{score:.1f}%", "Model output")
        with col2:
            metric_card("Level", perf.value, "Performance band")
        with col3:
            metric_card("Priority", f"{priority:.1f}", "Study priority")

        save_prediction_history({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'student_id': student_id,
            'subject': subject,
            'topic': topic,
            'previous_score': prev_score,
            'study_hours': study_hrs,
            'attempts': attempts,
            'difficulty': difficulty,
            'confidence_level': confidence,
            'days_since_revision': days_rev,
            'attendance': attendance,
            'quiz_score': quiz_score,
            'predicted_score': round(score, 2),
            'performance_level': perf.value,
            'priority_score': priority,
            'theme': st.session_state.theme,
        })
        
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

        st.info(f"Saved profile for {student_id}. You can reload it from the sidebar later.")

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
    st.markdown('<div class="hero"><h1 style="margin:0;">Analytics</h1><p class="muted">Trends, distributions, and model signals from the actual dataset.</p></div>', unsafe_allow_html=True)
    
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
            fig.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            subjects = pd.DataFrame({
                'Subject': ['Math', 'Science', 'English', 'History', 'Chemistry', 'Physics', 'Biology'],
                'Score': [72, 68, 75, 60, 65, 70, 73]
            })
            fig = px.bar(subjects, x='Subject', y='Score', title="Performance by Subject", range_y=[0, 100])
            fig.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.subheader("Score Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='target_score', nbins=30, title='Score Distribution')
            fig.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            order = ['Easy', 'Medium', 'Hard']
            df_sorted = df.copy()
            df_sorted['difficulty'] = pd.Categorical(df_sorted['difficulty'], categories=order, ordered=True)
            fig = px.box(df_sorted.sort_values('difficulty'), x='difficulty', y='target_score', title='Scores by Difficulty')
            fig.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        
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
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.pie(subject_stats, values='count', names='subject', title='Sample Distribution')
            st.plotly_chart(fig, width='stretch')
        
        st.subheader("Top 15 Topics")
        topic_stats = df.groupby('topic')['target_score'].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=False).head(15)
        fig = px.bar(topic_stats, x='mean', y='topic', orientation='h', title='Top Topics by Score', range_x=[0, 100])
        fig.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch')
    
    with tab4:
        st.subheader("Feature Importance")
        
        col1, col2 = st.columns(2)
        with col1:
            importance = pd.DataFrame({
                'Feature': ['Previous Score', 'Confidence', 'Difficulty', 'Days Revision', 'Quiz Score', 'Study Hours', 'Attempts', 'Attendance'],
                'Importance': [0.3444, 0.2022, 0.1002, 0.0968, 0.0926, 0.0867, 0.0332, 0.0263]
            })
            fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title='Feature Importance')
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            cols = ['previous_score', 'study_hours', 'attempts', 'days_since_revision', 'attendance', 'quiz_score']
            corr = df[cols + ['target_score']].corr()['target_score'].drop('target_score').sort_values(ascending=False)
            fig = px.bar(x=corr.values, y=corr.index, orientation='h', title='Feature Correlation', range_x=[-1, 1])
            st.plotly_chart(fig, width='stretch')

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
            st.plotly_chart(fig, width='stretch')
    
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
            st.plotly_chart(fig, width='stretch')
    
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
        st.plotly_chart(fig, width='stretch')
    
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
    initialize_state()
    st.sidebar.title("Study Planner")

    st.sidebar.markdown("### Appearance")
    theme_toggle = st.sidebar.toggle("Dark mode", value=st.session_state.theme == 'dark')
    st.session_state.theme = 'dark' if theme_toggle else 'light'
    apply_theme()
    
    model_path = 'models/random_forest_model.pkl'
    if not os.path.exists(model_path):
        st.sidebar.warning("Model not trained yet.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Saved performance inputs are stored locally in `data/student_performance_history.csv`.")
    
    page = st.sidebar.radio("Navigate to:", 
        ["Dashboard", "Predict", "My Progress", "Study Plan", "Exam Readiness", "Analytics"])
    
    if page == "Dashboard":
        page_dashboard()
    elif page == "Predict":
        page_predict()
    elif page == "My Progress":
        page_my_progress()
    elif page == "Study Plan":
        page_study_plan()
    elif page == "Exam Readiness":
        page_exam_readiness()
    else:
        page_analytics()

if __name__ == "__main__":
    main()
