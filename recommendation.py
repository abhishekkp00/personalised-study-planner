"""
Recommendation Engine for Study Planning
Generates priority scores and study recommendations
"""

import pandas as pd
import numpy as np
from enum import Enum

class PerformanceLevel(Enum):
    WEAK = "Weak"
    AVERAGE = "Average"
    STRONG = "Strong"

class RecommendationEngine:
    def __init__(self):
        """Initialize the recommendation engine"""
        self.difficulty_scores = {
            'Easy': 1,
            'Medium': 2,
            'Hard': 3
        }
        
    def classify_performance(self, predicted_score):
        """
        Classify performance based on predicted score
        
        Parameters:
        predicted_score: Predicted target score (0-100)
        
        Returns:
        PerformanceLevel enum
        """
        if predicted_score < 50:
            return PerformanceLevel.WEAK
        elif predicted_score < 75:
            return PerformanceLevel.AVERAGE
        else:
            return PerformanceLevel.STRONG
    
    def calculate_priority_score(self, predicted_score, difficulty, days_since_revision):
        """
        Calculate priority score for studying a topic
        
        Formula:
        priority_score = ((100 - predicted_score) * 0.5) + 
                        (difficulty_score * 10 * 0.3) + 
                        (days_since_revision * 0.2)
        
        Parameters:
        predicted_score: Predicted score (0-100)
        difficulty: 'Easy', 'Medium', or 'Hard'
        days_since_revision: Number of days since last revision
        
        Returns:
        float: Priority score (higher = should study first)
        """
        difficulty_score = self.difficulty_scores.get(difficulty, 2)
        
        priority_score = (
            ((100 - predicted_score) * 0.5) +
            (difficulty_score * 10 * 0.3) +
            (days_since_revision * 0.2)
        )
        
        return round(priority_score, 2)
    
    def calculate_recommended_hours(self, performance_level, difficulty):
        """
        Calculate recommended weekly study hours
        
        Base hours by performance level:
        - Weak: 6-8 hours/week
        - Average: 3-5 hours/week
        - Strong: 1-2 hours/week
        
        Adjusted based on difficulty:
        - Easy: no extra hours
        - Medium: +1 hour
        - Hard: +2 hours
        
        Parameters:
        performance_level: PerformanceLevel enum
        difficulty: 'Easy', 'Medium', or 'Hard'
        
        Returns:
        dict with min_hours, max_hours, recommended_hours
        """
        # Base hours
        base_hours = {
            PerformanceLevel.WEAK: (6, 8),
            PerformanceLevel.AVERAGE: (3, 5),
            PerformanceLevel.STRONG: (1, 2)
        }
        
        min_h, max_h = base_hours[performance_level]
        
        # Adjust based on difficulty
        difficulty_adjustment = {
            'Easy': 0,
            'Medium': 1,
            'Hard': 2
        }
        
        adjustment = difficulty_adjustment.get(difficulty, 0)
        
        min_hours = min_h + adjustment
        max_hours = max_h + adjustment
        recommended_hours = (min_hours + max_hours) / 2
        
        return {
            'min_hours': min_hours,
            'max_hours': max_hours,
            'recommended_hours': round(recommended_hours, 1)
        }
    
    def generate_recommendations(self, predicted_score, performance_level, 
                                difficulty, days_since_revision, quiz_score):
        """
        Generate text recommendations based on performance
        
        Parameters:
        predicted_score: Predicted score
        performance_level: PerformanceLevel enum
        difficulty: Difficulty level
        days_since_revision: Days since revision
        quiz_score: Current quiz score
        
        Returns:
        list of recommendation strings
        """
        recommendations = []
        
        if performance_level == PerformanceLevel.WEAK:
            recommendations.append("⚠️ WEAK TOPIC - Priority Intervention Needed")
            recommendations.append("• Revise fundamentals from scratch")
            recommendations.append("• Watch tutorial videos or reference materials")
            recommendations.append("• Solve beginner-level problems step by step")
            recommendations.append("• Take a quiz after 3 days of practice")
            recommendations.append("• Focus on understanding core concepts")
            recommendations.append("• Analyze mistakes and incorrect answers")
            
        elif performance_level == PerformanceLevel.AVERAGE:
            recommendations.append("📊 AVERAGE TOPIC - Improvement Needed")
            recommendations.append("• Review key concepts and formulas")
            recommendations.append("• Practice intermediate-level problems")
            recommendations.append("• Attempt quiz after 4-5 days")
            recommendations.append("• Compare your solutions with model answers")
            recommendations.append("• Identify specific weak areas within the topic")
            
        else:  # STRONG
            recommendations.append("✅ STRONG TOPIC - Revision Only")
            recommendations.append("• Use this for quick revision or practice")
            recommendations.append("• Attempt advanced problems for skill enhancement")
            recommendations.append("• Monthly revision recommended")
            recommendations.append("• Help others understand this topic")
            recommendations.append("• Explore real-world applications")
        
        # Additional recommendations based on other factors
        if difficulty == 'Hard':
            recommendations.append("\n💪 DIFFICULTY NOTE: This is a Hard topic")
            recommendations.append("• Allocate extra time for practice")
            recommendations.append("• Break concepts into smaller chunks")
        
        if days_since_revision > 30:
            recommendations.append("\n🔄 REVISION OVERDUE")
            recommendations.append(f"• Last revision was {days_since_revision} days ago")
            recommendations.append("• Immediate revision recommended to avoid forgetting")
        
        if quiz_score < 40:
            recommendations.append("\n❌ LOW QUIZ SCORE ALERT")
            recommendations.append("• Recent quiz performance is below expectations")
            recommendations.append("• Focus on foundational concepts first")
        
        return recommendations
    
    def generate_weekly_plan(self, student_topic_data, num_days=7):
        """
        Generate a weekly study plan for a student-topic pair
        
        Parameters:
        student_topic_data: dict with topic info and recommendations
        num_days: Number of days for the plan (default 7)
        
        Returns:
        list of daily study activities
        """
        performance_level = student_topic_data['performance_level']
        difficulty = student_topic_data['difficulty']
        recommended_hours = student_topic_data['recommended_hours']
        topic_name = student_topic_data['topic']
        
        weekly_plan = []
        
        if performance_level == PerformanceLevel.WEAK:
            plan = [
                f"Day 1 ({topic_name} - Fundamentals): 2 hours - Learn basic concepts",
                f"Day 2 (Foundational Problems): 2 hours - Solve easy practice problems",
                f"Day 3 (Concept Review): 1.5 hours - Revise difficult concepts",
                f"Day 4 (Problem Solving): 2 hours - Solve intermediate problems",
                f"Day 5 (Quiz & Analysis): 1.5 hours - Take a quiz and analyze mistakes",
                f"Day 6 (Mistake Review): 1.5 hours - Review and correct mistakes",
                f"Day 7 (Weekly Assessment): 2 hours - Attempt another assessment quiz"
            ]
        
        elif performance_level == PerformanceLevel.AVERAGE:
            plan = [
                f"Day 1 ({topic_name} - Review): 1.5 hours - Revise key concepts",
                f"Day 2 (Intermediate Problems): 1.5 hours - Practice medium-level problems",
                f"Day 3 (Concept Strengthening): 1 hour - Focus on weak areas",
                f"Day 4 (More Problems): 1.5 hours - Solve challenging medium problems",
                f"Day 5 (Quiz): 1 hour - Take assessment quiz",
                f"Day 6 (Analysis): 1 hour - Analyze quiz results",
                f"Day 7 (Revision): 1 hour - Revise and consolidate learning"
            ]
        
        else:  # STRONG
            plan = [
                f"Day 1-3 ({topic_name}): 0.5 hours/day - Quick revision or reference",
                f"Day 4 (Advanced Problems): 1 hour - Attempt advanced level problems",
                f"Day 5 (Application): 0.5 hours - Explore real-world applications",
                f"Day 6 (Peer Learning): 1 hour - Teach or discuss with peers",
                f"Day 7 (Monthly Check): 0.5 hours - Schedule next revision date"
            ]
        
        # Adjust for difficulty
        if difficulty == 'Hard':
            if performance_level != PerformanceLevel.STRONG:
                plan.append(f"\n⭐ Hard Topic Focus: Add 2 extra hours for deeper practice")
        
        return plan
    
    def generate_full_recommendation(self, student_data, predicted_score):
        """
        Generate complete recommendation for a student-topic pair
        
        Parameters:
        student_data: dict with student and topic information
        predicted_score: Model's predicted target score
        
        Returns:
        dict with all recommendations and plan
        """
        performance_level = self.classify_performance(predicted_score)
        
        study_hours_info = self.calculate_recommended_hours(
            performance_level, 
            student_data['difficulty']
        )
        
        priority_score = self.calculate_priority_score(
            predicted_score,
            student_data['difficulty'],
            student_data['days_since_revision']
        )
        
        recommendations = self.generate_recommendations(
            predicted_score,
            performance_level,
            student_data['difficulty'],
            student_data['days_since_revision'],
            student_data.get('quiz_score', 50)
        )
        
        weekly_plan = self.generate_weekly_plan({
            'topic': student_data['topic'],
            'performance_level': performance_level,
            'difficulty': student_data['difficulty'],
            'recommended_hours': study_hours_info['recommended_hours']
        })
        
        return {
            'topic': student_data['topic'],
            'subject': student_data['subject'],
            'predicted_score': round(predicted_score, 1),
            'performance_level': performance_level.value,
            'priority_score': priority_score,
            'difficulty': student_data['difficulty'],
            'recommended_hours_min': study_hours_info['min_hours'],
            'recommended_hours_max': study_hours_info['max_hours'],
            'recommended_hours': study_hours_info['recommended_hours'],
            'recommendations': recommendations,
            'weekly_plan': weekly_plan
        }

# Example usage function
def example_recommendation():
    """Example of how to use the recommendation engine"""
    engine = RecommendationEngine()
    
    # Sample data
    student_data = {
        'student_id': 'STU10001',
        'subject': 'DSA',
        'topic': 'Dynamic Programming',
        'difficulty': 'Hard',
        'days_since_revision': 45,
        'quiz_score': 35
    }
    
    predicted_score = 42.6  # From ML model
    
    # Generate recommendation
    result = engine.generate_full_recommendation(student_data, predicted_score)
    
    print("\n" + "="*70)
    print(f"PERSONALIZED STUDY RECOMMENDATION FOR: {result['topic']}")
    print("="*70)
    print(f"\nPredicted Score: {result['predicted_score']}%")
    print(f"Performance Level: {result['performance_level']}")
    print(f"Priority Score: {result['priority_score']}")
    print(f"Recommended Study Time: {result['recommended_hours']} hours/week")
    
    print("\nRECOMMENDATIONS:")
    for rec in result['recommendations']:
        print(rec)
    
    print("\nWEEKLY STUDY PLAN:")
    for day in result['weekly_plan']:
        print(day)
    print("="*70)

if __name__ == "__main__":
    example_recommendation()
