"""
Generate synthetic student performance dataset
Creates realistic academic patterns for 20,000 rows
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_dataset(num_samples=20000, random_state=42):
    """
    Generate synthetic student performance dataset with realistic patterns
    
    Parameters:
    num_samples: Number of records to generate (default 20000)
    random_state: For reproducibility
    
    Returns:
    DataFrame with student performance data
    """
    np.random.seed(random_state)
    
    # Define categorical values
    subjects = ['DSA', 'DBMS', 'OS', 'ML', 'CN', 'Python', 'Aptitude']
    
    subject_topics = {
        'DSA': ['Arrays', 'LinkedList', 'Stack', 'Queue', 'Tree', 'Graph', 'Sorting', 'Searching'],
        'DBMS': ['SQL', 'Normalization', 'ER-Model', 'Transactions', 'Indexing', 'Queries'],
        'OS': ['Processes', 'Threads', 'Deadlock', 'Memory', 'FileSystem', 'Scheduling'],
        'ML': ['Regression', 'Classification', 'Clustering', 'NeuralNets', 'DecisionTrees'],
        'CN': ['Routing', 'TCP-IP', 'DNS', 'Security', 'Protocols', 'Bandwidth'],
        'Python': ['Loops', 'Functions', 'OOP', 'Exceptions', 'Decorators', 'Generators'],
        'Aptitude': ['Percentage', 'Probability', 'Permutation', 'Algebra', 'Geometry']
    }
    
    difficulties = ['Easy', 'Medium', 'Hard']
    confidence_levels = ['Low', 'Medium', 'High']
    
    # Initialize data
    data = []
    num_students = 500  # Approximately 40 records per student (20000 / 500)
    
    for _ in range(num_samples):
        student_id = f"STU{np.random.randint(1001, 1001 + num_students):05d}"
        subject = np.random.choice(subjects)
        topic = np.random.choice(subject_topics[subject])
        difficulty = np.random.choice(difficulties)
        confidence_level = np.random.choice(confidence_levels)
        
        # Generate features with realistic patterns
        # Previous score strongly influences target score
        previous_score = np.random.normal(65, 20)
        previous_score = np.clip(previous_score, 0, 100)
        
        # Study hours - typically between 0-15 hours
        study_hours = np.random.exponential(2.5)
        study_hours = np.clip(study_hours, 0, 15)
        
        # Attempts/practice sessions - typically 1-20
        attempts = np.random.randint(1, 21)
        
        # Days since revision - 0 to 60 days
        days_since_revision = np.random.randint(0, 61)
        
        # Attendance/engagement - percentage
        attendance = np.random.normal(75, 15)
        attendance = np.clip(attendance, 0, 100)
        
        # Quiz score - influenced by previous_score
        quiz_score = previous_score + np.random.normal(5, 10)
        quiz_score = np.clip(quiz_score, 0, 100)
        
        # Calculate target score with realistic patterns
        target_score = previous_score * 0.35  # Previous performance
        target_score += quiz_score * 0.25     # Current quiz performance
        target_score += study_hours * 2       # 2 points per hour
        target_score += attempts * 0.5        # 0.5 points per attempt
        target_score += attendance * 0.15     # Attendance contribution
        target_score -= days_since_revision * 0.3  # Forgetting curve
        target_score += np.random.normal(0, 8)  # Random variance
        
        # Adjust based on difficulty
        if difficulty == 'Hard':
            target_score -= 15
        elif difficulty == 'Medium':
            target_score -= 5
        
        # Adjust based on confidence
        if confidence_level == 'High':
            target_score += 10
        elif confidence_level == 'Low':
            target_score -= 10
        
        # Ensure target score is in valid range
        target_score = np.clip(target_score, 0, 100)
        
        data.append({
            'student_id': student_id,
            'subject': subject,
            'topic': topic,
            'previous_score': round(previous_score, 2),
            'study_hours': round(study_hours, 2),
            'attempts': attempts,
            'difficulty': difficulty,
            'confidence_level': confidence_level,
            'days_since_revision': days_since_revision,
            'attendance': round(attendance, 2),
            'quiz_score': round(quiz_score, 2),
            'target_score': round(target_score, 2)
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    """Generate and save the dataset"""
    print("Generating synthetic student performance dataset...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate dataset
    df = generate_dataset(num_samples=20000)
    
    # Save to CSV
    output_path = '/home/abhishek/Projects/Personalized-Study-Planner/data/dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Saved to: {output_path}")
    
    # Display basic statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nNumerical Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nCategorical Distribution:")
    print(f"\nSubjects: {df['subject'].unique()}")
    print(f"Subject counts:\n{df['subject'].value_counts()}")
    print(f"\nDifficulty distribution:\n{df['difficulty'].value_counts()}")
    print(f"\nConfidence level distribution:\n{df['confidence_level'].value_counts()}")
    
    print("\n" + "="*60)
    print(f"Dataset ready at: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
