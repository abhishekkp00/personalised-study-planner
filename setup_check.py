"""
Utility functions for the Study Planner project
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'streamlit',
        'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def check_project_structure():
    """Verify project directory structure"""
    base_path = '/home/abhishek/Projects/Personalized-Study-Planner'
    required_dirs = ['data', 'models', 'scripts']
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            os.makedirs(dir_path, exist_ok=True)
    
    if missing_dirs:
        print(f"⚠️ Created missing directories: {missing_dirs}")
    else:
        print("✅ Project structure verified")
    
    return True

def check_dataset_exists():
    """Check if dataset exists"""
    dataset_path = '/home/abhishek/Projects/Personalized-Study-Planner/data/dataset.csv'
    
    if os.path.exists(dataset_path):
        # Check size
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"✅ Dataset found ({size_mb:.2f} MB)")
        return True
    else:
        print("❌ Dataset not found. Run: python generate_dataset.py")
        return False

def check_model_exists():
    """Check if trained model exists"""
    model_path = '/home/abhishek/Projects/Personalized-Study-Planner/models/random_forest_model.pkl'
    
    if os.path.exists(model_path):
        print("✅ Model found and ready")
        return True
    else:
        print("❌ Model not found. Run: python train_model.py")
        return False

def print_welcome():
    """Print welcome message"""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║   🎓 AI-BASED PERSONALIZED STUDY PLANNER               ║
    ║   Using Student Performance Prediction                 ║
    ╚════════════════════════════════════════════════════════╝
    """)

def print_setup_guide():
    """Print setup instructions"""
    print("""
    📋 SETUP INSTRUCTIONS:
    
    Step 1: Install Dependencies
    $ pip install -r requirements.txt
    
    Step 2: Generate Dataset
    $ python generate_dataset.py
    
    Step 3: Train Model
    $ python train_model.py
    
    Step 4: Run Web Application
    $ streamlit run app.py
    
    Your app will open at: http://localhost:8501
    """)

def print_quick_start():
    """Print quick start options"""
    print("""
    🚀 QUICK START:
    
    • Dashboard     : View overall analytics and insights
    • Predict       : Get personalized predictions
    • Study Plan    : Generate weekly study schedule
    • Analytics     : Explore data patterns
    
    🎯 Tips:
    - Enter your actual performance data for accurate predictions
    - Check your priority topics on the dashboard
    - Follow the recommended study hours for best results
    - Review the weekly plan at the start of each week
    """)

def run_setup_check():
    """Run complete setup verification"""
    print_welcome()
    
    print("\n🔍 Running Setup Verification...\n")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Dataset", check_dataset_exists),
        ("Model", check_model_exists)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\nChecking {check_name}...")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"Error checking {check_name}: {e}")
            results.append((check_name, False))
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20} {status}")
        all_passed = all_passed and result
    
    print("="*60)
    
    if all_passed:
        print("\n✨ System is ready to use!")
        print_quick_start()
    else:
        print("\n⚠️ Some checks failed. Please follow the setup guide:")
        print_setup_guide()
    
    return all_passed

if __name__ == "__main__":
    run_setup_check()
