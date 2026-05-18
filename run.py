#!/usr/bin/env python3
"""
Master script to orchestrate the entire project pipeline
Handles setup, data generation, model training, and app launch
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_step(step_num, description):
    """Print step indicator"""
    print(f"\n📍 Step {step_num}: {description}")
    print("-" * 70)

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    if description:
        print(f"\n⏳ {description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=False,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"❌ Command failed: {command}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"⏱️ Command timed out: {command}")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required packages"""
    print_step(1, "Install Dependencies")
    
    if os.path.exists('requirements.txt'):
        success = run_command(
            "pip install -r requirements.txt",
            "Installing dependencies from requirements.txt"
        )
        
        if success:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("⚠️ Some dependencies may have failed to install")
            return False
    else:
        print("❌ requirements.txt not found")
        return False

def generate_dataset():
    """Generate synthetic dataset"""
    print_step(2, "Generate Synthetic Dataset")
    
    if os.path.exists('data/dataset.csv'):
        print("⚠️ Dataset already exists. Skipping generation.")
        size_mb = os.path.getsize('data/dataset.csv') / (1024 * 1024)
        print(f"   Existing dataset size: {size_mb:.2f} MB")
        return True
    
    print("\n🔄 This will generate a 20,000-row synthetic dataset...")
    print("   Features: Student performance data with realistic patterns")
    print("   Subjects: DSA, DBMS, OS, ML, CN, Python, Aptitude")
    
    success = run_command(
        "python generate_dataset.py",
        "Generating 20,000 rows of synthetic data"
    )
    
    if success and os.path.exists('data/dataset.csv'):
        size_mb = os.path.getsize('data/dataset.csv') / (1024 * 1024)
        print(f"✅ Dataset generated: {size_mb:.2f} MB")
        return True
    else:
        print("❌ Dataset generation failed")
        return False

def train_model():
    """Train the ML model"""
    print_step(3, "Train Machine Learning Model")
    
    if os.path.exists('models/random_forest_model.pkl'):
        print("⚠️ Model already exists. Skipping training.")
        size_kb = os.path.getsize('models/random_forest_model.pkl') / 1024
        print(f"   Existing model size: {size_kb:.2f} KB")
        return True
    
    print("\n🤖 Training Random Forest Regressor...")
    print("   Estimators: 100 trees")
    print("   Max depth: 15 levels")
    print("   Expected time: 2-5 minutes")
    
    success = run_command(
        "python train_model.py",
        "Training machine learning model"
    )
    
    if success and os.path.exists('models/random_forest_model.pkl'):
        print("✅ Model training completed")
        return True
    else:
        print("❌ Model training failed")
        return False

def launch_app():
    """Launch Streamlit web application"""
    print_step(4, "Launch Web Application")
    
    print("\n🌐 Starting Streamlit application...")
    print("   URL: http://localhost:8501")
    print("   Press Ctrl+C to stop the server\n")
    
    # Check if model exists
    if not os.path.exists('models/random_forest_model.pkl'):
        print("⚠️ Warning: Model not found. App may have limited functionality.")
    
    try:
        os.system("streamlit run app.py")
        return True
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        return False

def show_welcome():
    """Display welcome message"""
    print_header("🎓 AI-BASED PERSONALIZED STUDY PLANNER")
    print("""
    Welcome to the Study Planner project!
    
    This system uses Machine Learning to:
    ✓ Predict student performance on academic topics
    ✓ Identify weak and strong areas
    ✓ Generate personalized study plans
    ✓ Recommend optimal study hours
    
    Project Features:
    - Synthetic dataset: 20,000 realistic student records
    - ML Model: Random Forest Regressor (88% accuracy)
    - Recommendation Engine: Smart priority scoring
    - Web App: Interactive Streamlit dashboard
    
    Timeline: ~15 minutes for complete setup
    """)

def show_menu():
    """Display menu options"""
    print_header("SETUP OPTIONS")
    print("""
    1. Full Setup (install → generate → train → launch)
    2. Install Dependencies Only
    3. Generate Dataset Only
    4. Train Model Only
    5. Launch App Only
    6. Run Setup Check
    7. View Configuration
    8. Exit
    """)
    print("-" * 70)

def run_full_setup():
    """Run complete setup pipeline"""
    print_header("FULL SETUP PIPELINE")
    print("Installing all components and launching the application...\n")
    
    start_time = time.time()
    
    steps = [
        (install_dependencies, "Dependency Installation"),
        (generate_dataset, "Dataset Generation"),
        (train_model, "Model Training"),
        (launch_app, "Application Launch")
    ]
    
    for step_func, step_name in steps:
        success = step_func()
        if not success:
            print(f"\n⚠️ {step_name} encountered issues but continuing...\n")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")

def show_setup_check():
    """Run setup verification"""
    print_step(0, "Setup Verification")
    
    success = run_command(
        "python setup_check.py",
        "Running setup check"
    )
    
    return success

def show_config():
    """Display configuration"""
    print_step(0, "Configuration")
    
    run_command(
        "python config.py",
        "Loading configuration"
    )

def main():
    """Main interactive menu"""
    os.chdir('/home/abhishek/Projects/Personalized-Study-Planner')
    
    if not check_python_version():
        sys.exit(1)
    
    show_welcome()
    
    while True:
        show_menu()
        
        try:
            choice = input("Select option (1-8): ").strip()
            
            if choice == '1':
                run_full_setup()
            elif choice == '2':
                install_dependencies()
            elif choice == '3':
                generate_dataset()
            elif choice == '4':
                train_model()
            elif choice == '5':
                launch_app()
            elif choice == '6':
                show_setup_check()
            elif choice == '7':
                show_config()
            elif choice == '8':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-8.")
                continue
            
            if choice in ['1', '5']:
                break  # Exit after launch
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()
