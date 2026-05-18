"""
Configuration settings for the Study Planner project
"""

import os

# Project paths
BASE_PATH = '/home/abhishek/Projects/Personalized-Study-Planner'
DATA_PATH = os.path.join(BASE_PATH, 'data')
MODELS_PATH = os.path.join(BASE_PATH, 'models')
SCRIPTS_PATH = os.path.join(BASE_PATH, 'scripts')

# Dataset configuration
DATASET_FILE = os.path.join(DATA_PATH, 'dataset.csv')
DATASET_SIZE = 20000  # Number of rows to generate

# Subjects and topics
SUBJECTS = ['DSA', 'DBMS', 'OS', 'ML', 'CN', 'Python', 'Aptitude']

SUBJECT_TOPICS = {
    'DSA': ['Arrays', 'LinkedList', 'Stack', 'Queue', 'Tree', 'Graph', 'Sorting', 'Searching', 
            'DynamicProgramming', 'Greedy', 'Recursion', 'BitManipulation'],
    'DBMS': ['SQL', 'Normalization', 'ERModel', 'Transactions', 'Indexing', 'Queries', 
             'JoinTypes', 'Aggregation', 'StorageStructures', 'RecoveryAlgorithms'],
    'OS': ['Processes', 'Threads', 'Deadlock', 'Memory', 'FileSystem', 'Scheduling', 
           'IPC', 'Synchronization', 'VirtualMemory', 'PageReplacement'],
    'ML': ['Regression', 'Classification', 'Clustering', 'NeuralNets', 'DecisionTrees', 
           'SVM', 'Ensemble', 'NLP', 'ComputerVision', 'Validation'],
    'CN': ['Routing', 'TCPIP', 'DNS', 'Security', 'Protocols', 'Bandwidth', 
           'Congestion', 'ErrorHandling', 'Authentication', 'NAT'],
    'Python': ['Loops', 'Functions', 'OOP', 'Exceptions', 'Decorators', 'Generators', 
               'Comprehensions', 'FileIO', 'Modules', 'Testing'],
    'Aptitude': ['Percentage', 'Probability', 'Permutation', 'Algebra', 'Geometry', 
                 'TimeSpeeds', 'LogicProblems', 'SeriesPatterns', 'Ratios', 'Averages']
}

DIFFICULTIES = ['Easy', 'Medium', 'Hard']
CONFIDENCE_LEVELS = ['Low', 'Medium', 'High']

# Model configuration
MODEL_TYPE = 'random_forest'  # Options: 'random_forest', 'gradient_boosting', 'linear'
MODEL_FILE = os.path.join(MODELS_PATH, 'random_forest_model.pkl')
ENCODERS_FILE = os.path.join(MODELS_PATH, 'encoders.pkl')
FEATURE_NAMES_FILE = os.path.join(MODELS_PATH, 'feature_names.pkl')
FEATURE_IMPORTANCE_FILE = os.path.join(MODELS_PATH, 'feature_importance.pkl')

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 15
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 5
RF_RANDOM_STATE = 42

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Performance classification thresholds
WEAK_THRESHOLD = 50      # Score < 50 is Weak
AVERAGE_THRESHOLD = 75   # 50 <= Score < 75 is Average
# Score >= 75 is Strong

# Study hours recommendation
STUDY_HOURS = {
    'Weak': {
        'min': 6,
        'max': 8,
        'difficulty_adjustment': {'Easy': 0, 'Medium': 1, 'Hard': 2}
    },
    'Average': {
        'min': 3,
        'max': 5,
        'difficulty_adjustment': {'Easy': 0, 'Medium': 1, 'Hard': 2}
    },
    'Strong': {
        'min': 1,
        'max': 2,
        'difficulty_adjustment': {'Easy': 0, 'Medium': 1, 'Hard': 2}
    }
}

# Priority score formula weights
PRIORITY_SCORE_WEIGHTS = {
    'performance_inverse': 0.5,    # Weight for (100 - predicted_score)
    'difficulty': 0.3,             # Weight for difficulty component
    'days_revision': 0.2           # Weight for days since revision
}

# Difficulty scores
DIFFICULTY_SCORES = {
    'Easy': 1,
    'Medium': 2,
    'Hard': 3
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    'layout': 'wide',
    'theme': 'light',
    'max_upload_size': 200
}

# Feature configuration
FEATURE_COLUMNS = [
    'student_id',
    'subject',
    'topic',
    'previous_score',
    'study_hours',
    'attempts',
    'difficulty',
    'confidence_level',
    'days_since_revision',
    'attendance',
    'quiz_score',
    'target_score'
]

NUMERIC_FEATURES = [
    'previous_score',
    'study_hours',
    'attempts',
    'days_since_revision',
    'attendance',
    'quiz_score'
]

CATEGORICAL_FEATURES = [
    'subject',
    'topic',
    'difficulty',
    'confidence_level'
]

TARGET_FEATURE = 'target_score'

# Data generation parameters
DATA_GEN_NUM_STUDENTS = 500  # Approximately 40 samples per student
DATA_GEN_RANDOM_STATE = 42

# Common recommendations templates
RECOMMENDATIONS = {
    'Weak': [
        "Revise fundamentals from scratch",
        "Watch tutorial videos or reference materials",
        "Solve beginner-level problems step by step",
        "Take a quiz after 3 days of practice",
        "Focus on understanding core concepts",
        "Analyze mistakes and incorrect answers",
    ],
    'Average': [
        "Review key concepts and formulas",
        "Practice intermediate-level problems",
        "Attempt quiz after 4-5 days",
        "Compare your solutions with model answers",
        "Identify specific weak areas within the topic",
    ],
    'Strong': [
        "Use this for quick revision or practice",
        "Attempt advanced problems for skill enhancement",
        "Monthly revision recommended",
        "Help others understand this topic",
        "Explore real-world applications",
    ]
}

# Dashboard settings
DASHBOARD_METRICS = [
    'total_topics',
    'weak_topics',
    'average_topics',
    'strong_topics',
    'average_score',
    'highest_priority_topic'
]

# Logging configuration
LOG_FILE = os.path.join(BASE_PATH, 'app.log')
LOG_LEVEL = 'INFO'

# Feature importance threshold (show features with importance > threshold)
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# Number of top topics to display in dashboard
TOP_TOPICS_COUNT = 10

# Analytics configuration
ANALYTICS_SAMPLE_SIZE = 5000  # Use sample for performance
ANALYTICS_CHART_HEIGHT = 400
ANALYTICS_CHART_WIDTH = 600

def get_config(key, default=None):
    """Get configuration value by key"""
    return globals().get(key, default)

def print_config():
    """Print all configuration settings"""
    print("\n" + "="*60)
    print("PROJECT CONFIGURATION")
    print("="*60)
    
    for key, value in sorted(globals().items()):
        if not key.startswith('_') and not callable(value):
            print(f"{key:30} = {value}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    print_config()
