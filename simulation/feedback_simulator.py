"""
Feedback simulator for the AeroNexus AI platform.
Generates realistic passenger feedback with varying sentiment for testing NLP models.
"""

import random
import numpy as np
from datetime import datetime, timedelta

class FeedbackSimulator:
    def __init__(self, seed=42):
        """Initialize feedback simulator with templates and patterns"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Feedback templates by sentiment and airport
        self.feedback_templates = {
            'positive': {
                'general': [
                    "Excellent airport experience! {specific_praise}",
                    "Really impressed with {feature}. {additional_comment}",
                    "Outstanding service throughout my journey. {specific_praise}",
                    "Great facilities and {feature}. Will definitely recommend!",
                    "Smooth and efficient experience. {specific_praise}",
                    "Love the {feature}! {additional_comment}",
                    "Perfect airport for travelers. {specific_praise}",
                    "Amazing experience from arrival to departure. {additional_comment}"
                ],
                'shopping': [
                    "The shopping options are fantastic! {specific_praise}",
                    "Great variety of stores and {feature}. {additional_comment}",
                    "Loved the retail experience here. {specific_praise}",
                    "Perfect place to shop before my flight. {additional_comment}",
                    "Excellent shopping facilities with {feature}.",
                    "The {feature} made shopping so convenient!"
                ],
                'food': [
                    "The food options are incredible! {specific_praise}",
                    "Delicious meals and {feature}. {additional_comment}",
                    "Great restaurants with authentic flavors. {specific_praise}",
                    "Perfect dining experience before my flight. {additional_comment}",
                    "The {feature} offers amazing variety!",
                    "Outstanding food quality and service."
                ],
                'service': [
                    "Staff were extremely helpful and professional. {specific_praise}",
                    "Excellent customer service throughout. {additional_comment}",
                    "The team went above and beyond to help. {specific_praise}",
                    "Friendly and efficient staff everywhere. {additional_comment}",
                    "Customer service is top-notch here!",
                    "Staff made my journey so much easier."
                ]
            },
            'negative': {
                'general': [
                    "Disappointed with the overall experience. {specific_complaint}",
                    "Several issues with {problem_area}. {additional_complaint}",
                    "Not impressed with the facilities. {specific_complaint}",
                    "Poor experience, especially {problem_area}. {additional_complaint}",
                    "Many problems during my visit. {specific_complaint}",
                    "Needs significant improvement in {problem_area}.",
                    "Frustrating experience overall. {additional_complaint}",
                    "Expected better from such a {airport_tier} airport."
                ],
                'shopping': [
                    "Limited shopping options and {problem_area}. {additional_complaint}",
                    "Overpriced items with poor {problem_area}. {specific_complaint}",
                    "Shopping experience was disappointing. {additional_complaint}",
                    "Not enough variety in stores. {specific_complaint}",
                    "The retail area needs major improvement.",
                    "Poor shopping facilities and expensive prices."
                ],
                'food': [
                    "Food quality is poor and {problem_area}. {additional_complaint}",
                    "Limited food options with {problem_area}. {specific_complaint}",
                    "Overpriced meals with poor service. {additional_complaint}",
                    "Food court needs significant improvement. {specific_complaint}",
                    "Terrible dining experience and long waits.",
                    "Poor food quality for the price paid."
                ],
                'service': [
                    "Staff were unhelpful and {problem_area}. {additional_complaint}",
                    "Poor customer service throughout. {specific_complaint}",
                    "Rude staff and no assistance provided. {additional_complaint}",
                    "Service quality is unacceptable. {specific_complaint}",
                    "Staff need better training and attitude.",
                    "No help when needed most."
                ]
            },
            'neutral': {
                'general': [
                    "Average experience overall. {neutral_observation}",
                    "Decent facilities but {improvement_area} could be better. {neutral_observation}",
                    "Standard airport experience. {neutral_observation}",
                    "It's okay, nothing special about {feature}. {improvement_area} needs work.",
                    "Fair experience with room for improvement in {improvement_area}.",
                    "Acceptable but {improvement_area} could be enhanced."
                ],
                'mixed': [
                    "Good {positive_aspect} but {negative_aspect} was disappointing.",
                    "Nice {positive_aspect} although {negative_aspect} needs improvement.",
                    "Satisfied with {positive_aspect} but not {negative_aspect}.",
                    "The {positive_aspect} is great but {negative_aspect} is poor.",
                    "Mixed experience - loved {positive_aspect}, hated {negative_aspect}."
                ]
            }
        }
        
        # Context-specific phrases
        self.specific_elements = {
            'DEL': {
                'positive_features': [
                    'modern terminal design', 'digital signage system', 'efficient security process',
                    'wide variety of shops', 'excellent food courts', 'comfortable seating areas',
                    'good WiFi connectivity', 'helpful airport staff', 'clean facilities',
                    'convenient parking', 'metro connectivity', 'duty-free shopping'
                ],
                'negative_areas': [
                    'long security queues', 'crowded terminals', 'expensive food prices',
                    'limited seating near gates', 'confusing signage', 'poor parking availability',
                    'slow immigration process', 'noisy environment', 'inadequate charging points',
                    'overpriced retail items', 'long walking distances', 'air conditioning issues'
                ],
                'neutral_aspects': [
                    'terminal facilities', 'shopping options', 'food variety',
                    'seating arrangements', 'digital displays', 'ground transportation'
                ]
            },
            'JAI': {
                'positive_features': [
                    'authentic Rajasthani cuisine', 'beautiful architecture', 'friendly local staff',
                    'good souvenir shops', 'traditional handicrafts', 'cultural ambiance',
                    'quick check-in process', 'less crowded terminals', 'affordable prices',
                    'convenient size', 'local art displays', 'heritage theme'
                ],
                'negative_areas': [
                    'limited international food options', 'fewer retail stores',
                    'basic facilities', 'limited shopping variety', 'small waiting areas',
                    'fewer charging stations', 'limited WiFi coverage', 'basic amenities',
                    'fewer flight connections', 'limited parking space', 'basic dining options'
                ],
                'neutral_aspects': [
                    'local restaurants', 'craft stores', 'waiting areas',
                    'basic amenities', 'regional connectivity', 'local culture theme'
                ]
            }
        }
        
        # Passenger segment characteristics
        self.segment_preferences = {
            'business': {
                'priorities': ['efficiency', 'premium services', 'connectivity', 'time-saving'],
                'complaints': ['delays', 'poor service', 'lack of premium facilities', 'inefficiency']
            },
            'leisure': {
                'priorities': ['value for money', 'variety', 'comfort', 'experience'],
                'complaints': ['high prices', 'limited options', 'poor facilities', 'unfriendly service']
            }
        }
    
    def generate_feedback(self, airport_code, sentiment_bias='mixed', passenger_segment=None):
        """Generate realistic feedback for specified airport and sentiment"""
        
        # Determine sentiment
        if sentiment_bias == 'mixed':
            sentiment = random.choices(
                ['positive', 'negative', 'neutral'],
                weights=[0.4, 0.3, 0.3]  # Slightly positive bias
            )[0]
        else:
            sentiment = sentiment_bias
        
        # Determine passenger segment if not provided
        if passenger_segment is None:
            if airport_code == 'DEL':
                passenger_segment = random.choices(['business', 'leisure'], weights=[0.55, 0.45])[0]
            else:  # JAI
                passenger_segment = random.choices(['business', 'leisure'], weights=[0.30, 0.70])[0]
        
        # Generate passenger ID
        passenger_id = f"{airport_code}_{random.randint(1000, 9999)}"
        
        # Select feedback category
        if sentiment == 'neutral':
            category = random.choice(['general', 'mixed'])
        else:
            category = random.choices(
                ['general', 'shopping', 'food', 'service'],
                weights=[0.4, 0.2, 0.2, 0.2]
            )[0]
        
        # Get base template
        templates = self.feedback_templates[sentiment][category]
        base_template = random.choice(templates)
        
        # Generate specific content
        feedback_text = self._populate_template(
            base_template, airport_code, sentiment, category, passenger_segment
        )
        
        # Calculate sentiment score and NPS
        sentiment_score = self._calculate_sentiment_score(sentiment, feedback_text)
        nps_score = self._sentiment_to_nps(sentiment_score)
        nps_category = self._get_nps_category(nps_score)
        
        # Generate timestamp (last 30 days)
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Determine feedback channel
        channel = random.choices(
            ['app', 'survey', 'email', 'kiosk'],
            weights=[0.3, 0.4, 0.2, 0.1]
        )[0]
        
        return {
            'feedback_id': f"FB_{random.randint(100000, 999999)}",
            'passenger_id': passenger_id,
            'airport_code': airport_code,
            'feedback_text': feedback_text,
            'sentiment_score': round(sentiment_score, 3),
            'nps_score': nps_score,
            'category': nps_category,
            'timestamp': timestamp,
            'channel': channel
        }
    
    def _populate_template(self, template, airport_code, sentiment, category, passenger_segment):
        """Populate template with specific content"""
        
        airport_elements = self.specific_elements[airport_code]
        
        # Replacement variables
        replacements = {}
        
        if sentiment == 'positive':
            replacements['specific_praise'] = random.choice([
                f"The {random.choice(airport_elements['positive_features'])} exceeded expectations",
                f"Particularly loved the {random.choice(airport_elements['positive_features'])}",
                f"Great experience with {random.choice(airport_elements['positive_features'])}"
            ])
            
            replacements['feature'] = random.choice(airport_elements['positive_features'])
            
            replacements['additional_comment'] = random.choice([
                "Made my journey so much smoother",
                "Will definitely use this airport again",
                "Highly recommend to other travelers",
                "Perfect for both business and leisure travelers",
                "Sets a great example for other airports"
            ])
        
        elif sentiment == 'negative':
            replacements['specific_complaint'] = random.choice([
                f"The {random.choice(airport_elements['negative_areas'])} was particularly frustrating",
                f"Major issues with {random.choice(airport_elements['negative_areas'])}",
                f"Disappointed by {random.choice(airport_elements['negative_areas'])}"
            ])
            
            replacements['problem_area'] = random.choice(airport_elements['negative_areas'])
            
            replacements['additional_complaint'] = random.choice([
                "Hope they improve these issues soon",
                "Made my travel experience stressful",
                "Would consider other airports next time",
                "Not what I expected from this airport",
                "Needs immediate attention from management"
            ])
            
            replacements['airport_tier'] = "major" if airport_code == 'DEL' else "regional"
        
        else:  # neutral
            replacements['neutral_observation'] = random.choice([
                "Nothing particularly stands out",
                "Meets basic expectations",
                "Standard for this type of airport",
                "Could be better with some improvements"
            ])
            
            replacements['improvement_area'] = random.choice(airport_elements['neutral_aspects'])
            replacements['feature'] = random.choice(airport_elements['neutral_aspects'])
            
            if category == 'mixed':
                replacements['positive_aspect'] = random.choice(airport_elements['positive_features'])
                replacements['negative_aspect'] = random.choice(airport_elements['negative_areas'])
        
        # Apply segment-specific modifications
        if passenger_segment == 'business':
            business_priorities = self.segment_preferences['business']['priorities']
            if any(priority in template.lower() for priority in business_priorities):
                if sentiment == 'positive':
                    replacements['additional_comment'] = "Perfect for business travelers"
                elif sentiment == 'negative':
                    replacements['additional_complaint'] = "Impacts business travel efficiency"
        
        # Replace placeholders in template
        result = template
        for key, value in replacements.items():
            placeholder = '{' + key + '}'
            if placeholder in result:
                result = result.replace(placeholder, value)
        
        # Clean up any remaining placeholders
        import re
        result = re.sub(r'\{[^}]*\}', '', result)
        result = re.sub(r'\s+', ' ', result).strip()
        
        # Add passenger segment context occasionally
        if random.random() < 0.3:
            if passenger_segment == 'business':
                result += " As a frequent business traveler, this matters a lot."
            else:
                result += " Great for families and leisure travelers."
        
        return result
    
    def _calculate_sentiment_score(self, sentiment_type, text):
        """Calculate numerical sentiment score from text and type"""
        
        base_scores = {
            'positive': 0.6,
            'negative': -0.4,
            'neutral': 0.0
        }
        
        base_score = base_scores[sentiment_type]
        
        # Add some variation based on text characteristics
        text_lower = text.lower()
        
        # Positive words boost
        positive_words = ['excellent', 'amazing', 'perfect', 'outstanding', 'love', 'great', 'fantastic']
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        # Negative words decrease
        negative_words = ['terrible', 'awful', 'poor', 'disappointing', 'worst', 'hate', 'horrible']
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Adjust score
        adjustment = (positive_count * 0.1) - (negative_count * 0.1)
        final_score = np.clip(base_score + adjustment + random.uniform(-0.1, 0.1), -1.0, 1.0)
        
        return final_score
    
    def _sentiment_to_nps(self, sentiment_score):
        """Convert sentiment score to NPS score (0-10)"""
        
        # Map sentiment [-1, 1] to NPS [0, 10]
        base_nps = (sentiment_score + 1) * 5
        
        # Add some randomness
        nps_with_noise = base_nps + random.uniform(-1, 1)
        
        # Ensure bounds
        return max(0, min(10, round(nps_with_noise, 1)))
    
    def _get_nps_category(self, nps_score):
        """Categorize NPS score into promoter/passive/detractor"""
        
        if nps_score >= 9:
            return 'promoter'
        elif nps_score >= 7:
            return 'passive'
        else:
            return 'detractor'
    
    def generate_bulk_feedback(self, airport_code, count=100, sentiment_distribution=None):
        """Generate multiple feedback entries for an airport"""
        
        if sentiment_distribution is None:
            sentiment_distribution = {'positive': 0.4, 'negative': 0.3, 'neutral': 0.3}
        
        feedback_list = []
        
        for _ in range(count):
            # Select sentiment based on distribution
            sentiment = random.choices(
                list(sentiment_distribution.keys()),
                weights=list(sentiment_distribution.values())
            )[0]
            
            feedback = self.generate_feedback(airport_code, sentiment)
            feedback_list.append(feedback)
        
        return feedback_list
    
    def simulate_feedback_trends(self, airport_code, days=30):
        """Simulate feedback trends over time with realistic patterns"""
        
        feedback_list = []
        
        for day in range(days):
            current_date = datetime.now() - timedelta(days=days - day - 1)
            
            # Vary daily sentiment (weekends slightly more positive)
            if current_date.weekday() >= 5:  # Weekend
                sentiment_weights = [0.5, 0.25, 0.25]  # More positive
            else:
                sentiment_weights = [0.35, 0.35, 0.30]  # Balanced
            
            # Generate 3-10 feedback entries per day
            daily_count = random.randint(3, 10)
            
            for _ in range(daily_count):
                sentiment = random.choices(
                    ['positive', 'negative', 'neutral'],
                    weights=sentiment_weights
                )[0]
                
                feedback = self.generate_feedback(airport_code, sentiment)
                
                # Adjust timestamp to specific day
                feedback['timestamp'] = current_date.replace(
                    hour=random.randint(6, 22),
                    minute=random.randint(0, 59),
                    second=0,
                    microsecond=0
                )
                
                feedback_list.append(feedback)
        
        return feedback_list

if __name__ == "__main__":
    # Test the feedback simulator
    simulator = FeedbackSimulator()
    
    # Generate sample feedback
    del_feedback = simulator.generate_feedback('DEL', 'positive')
    jai_feedback = simulator.generate_feedback('JAI', 'negative')
    
    print("DEL Positive Feedback:")
    print(del_feedback['feedback_text'])
    print(f"Sentiment: {del_feedback['sentiment_score']:.3f}, NPS: {del_feedback['nps_score']}")
    
    print("\nJAI Negative Feedback:")
    print(jai_feedback['feedback_text'])
    print(f"Sentiment: {jai_feedback['sentiment_score']:.3f}, NPS: {jai_feedback['nps_score']}")
