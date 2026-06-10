"""
Natural Language Processing model for sentiment analysis and NPS prediction.
Processes passenger feedback to derive sentiment scores and map to NPS categories.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import pickle
import os
from datetime import datetime
from collections import Counter

class SentimentNLPAnalyzer:
    def __init__(self):
        self.sentiment_threshold_promoter = 0.3
        self.sentiment_threshold_detractor = -0.1
        
        # Keywords for different aspects
        self.aspect_keywords = {
            'service': ['service', 'staff', 'help', 'assistance', 'friendly', 'rude', 'polite'],
            'shopping': ['shop', 'store', 'retail', 'buy', 'purchase', 'expensive', 'cheap', 'price'],
            'food': ['food', 'restaurant', 'cafe', 'meal', 'hungry', 'delicious', 'tasty', 'bland'],
            'facility': ['clean', 'dirty', 'toilet', 'washroom', 'wifi', 'charging', 'seating'],
            'navigation': ['sign', 'direction', 'lost', 'confusing', 'clear', 'easy', 'difficult'],
            'wait_time': ['queue', 'wait', 'fast', 'slow', 'quick', 'delay', 'time', 'rush']
        }
        
        # Aspect importance weights for overall satisfaction
        self.aspect_weights = {
            'service': 0.25,
            'shopping': 0.20,
            'food': 0.20,
            'facility': 0.15,
            'navigation': 0.10,
            'wait_time': 0.10
        }
        
        # Enhanced keyword mappings for better sentiment detection
        self.positive_keywords = [
            'excellent', 'outstanding', 'amazing', 'perfect', 'wonderful', 'fantastic',
            'great', 'good', 'nice', 'pleasant', 'helpful', 'friendly', 'clean',
            'fast', 'quick', 'easy', 'convenient', 'satisfied', 'happy', 'love'
        ]
        
        self.negative_keywords = [
            'terrible', 'awful', 'horrible', 'worst', 'bad', 'poor', 'disappointing',
            'slow', 'dirty', 'rude', 'unfriendly', 'expensive', 'confusing', 'difficult',
            'frustrated', 'angry', 'hate', 'annoying', 'crowded', 'noisy'
        ]
        
        self.performance_history = []
    
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob with enhancements"""
        if not text or not isinstance(text, str):
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0
            }
        
        cleaned_text = self.preprocess_text(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(cleaned_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Enhance with keyword-based analysis
        enhanced_polarity = self._enhance_sentiment_with_keywords(cleaned_text, polarity)
        
        # Determine sentiment label
        if enhanced_polarity > self.sentiment_threshold_promoter:
            sentiment_label = 'positive'
        elif enhanced_polarity < self.sentiment_threshold_detractor:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Calculate confidence based on polarity strength and subjectivity
        confidence = min(abs(enhanced_polarity) * (1 - subjectivity) + 0.1, 1.0)
        
        return {
            'polarity': enhanced_polarity,
            'subjectivity': subjectivity,
            'sentiment_label': sentiment_label,
            'confidence': confidence
        }
    
    def _enhance_sentiment_with_keywords(self, text, base_polarity):
        """Enhance sentiment analysis with domain-specific keywords"""
        words = text.split()
        
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        
        # Calculate keyword-based adjustment
        keyword_score = (positive_count - negative_count) / max(len(words), 1) * 2
        
        # Weighted combination of TextBlob and keyword-based scores
        enhanced_polarity = (base_polarity * 0.7) + (keyword_score * 0.3)
        
        # Clip to [-1, 1] range
        return np.clip(enhanced_polarity, -1, 1)
    
    def analyze_aspects(self, text):
        """Analyze specific aspects mentioned in the feedback"""
        if not text or not isinstance(text, str):
            return {}
        
        cleaned_text = self.preprocess_text(text)
        words = cleaned_text.split()
        
        aspect_sentiments = {}
        
        for aspect, keywords in self.aspect_keywords.items():
            # Check if aspect is mentioned
            aspect_mentions = sum(1 for word in words if word in keywords)
            
            if aspect_mentions > 0:
                # Extract sentences containing aspect keywords
                sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
                relevant_sentences = [
                    s for s in sentences 
                    if any(keyword in s for keyword in keywords)
                ]
                
                if relevant_sentences:
                    # Analyze sentiment of relevant sentences
                    aspect_text = '. '.join(relevant_sentences)
                    aspect_sentiment = self.analyze_sentiment(aspect_text)
                    
                    aspect_sentiments[aspect] = {
                        'sentiment': aspect_sentiment['polarity'],
                        'mentions': aspect_mentions,
                        'confidence': aspect_sentiment['confidence']
                    }
        
        return aspect_sentiments
    
    def sentiment_to_nps(self, sentiment_score, confidence=1.0):
        """Convert sentiment score to NPS score and category"""
        # Map sentiment [-1, 1] to NPS [0, 10]
        base_nps = (sentiment_score + 1) * 5
        
        # Add some randomness based on confidence
        noise = np.random.normal(0, (1 - confidence) * 2)
        nps_score = np.clip(base_nps + noise, 0, 10)
        
        # Determine NPS category
        if nps_score >= 9:
            nps_category = 'promoter'
        elif nps_score >= 7:
            nps_category = 'passive'
        else:
            nps_category = 'detractor'
        
        return {
            'nps_score': round(nps_score, 1),
            'nps_category': nps_category,
            'confidence': confidence
        }
    
    def calculate_nps_from_feedback(self, feedback_list):
        """Calculate overall NPS from a list of feedback"""
        if not feedback_list:
            return {
                'nps': 0,
                'promoters': 0,
                'passives': 0,
                'detractors': 0,
                'total_responses': 0
            }
        
        categories = {'promoter': 0, 'passive': 0, 'detractor': 0}
        
        for feedback in feedback_list:
            if isinstance(feedback, dict) and 'feedback_text' in feedback:
                text = feedback['feedback_text']
            else:
                text = str(feedback)
            
            sentiment = self.analyze_sentiment(text)
            nps_result = self.sentiment_to_nps(sentiment['polarity'], sentiment['confidence'])
            categories[nps_result['nps_category']] += 1
        
        total = sum(categories.values())
        if total == 0:
            return {
                'nps': 0,
                'promoters': 0,
                'passives': 0,
                'detractors': 0,
                'total_responses': 0
            }
        
        # Calculate NPS score
        promoter_pct = categories['promoter'] / total * 100
        detractor_pct = categories['detractor'] / total * 100
        nps = promoter_pct - detractor_pct
        
        return {
            'nps': round(nps, 1),
            'promoters': categories['promoter'],
            'passives': categories['passive'],
            'detractors': categories['detractor'],
            'total_responses': total,
            'promoter_percentage': round(promoter_pct, 1),
            'detractor_percentage': round(detractor_pct, 1)
        }
    
    def analyze_feedback_trends(self, feedback_df):
        """Analyze trends in feedback over time"""
        if feedback_df.empty:
            return {}
        
        # Ensure timestamp column exists
        if 'timestamp' not in feedback_df.columns:
            feedback_df['timestamp'] = datetime.now()
        
        # Convert timestamp to datetime if it's not already
        feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
        
        # Add derived columns
        feedback_df['date'] = feedback_df['timestamp'].dt.date
        feedback_df['hour'] = feedback_df['timestamp'].dt.hour
        
        trends = {}
        
        # Daily trend
        daily_sentiment = feedback_df.groupby('date').apply(
            lambda x: np.mean([
                self.analyze_sentiment(text)['polarity'] 
                for text in x['feedback_text'] if text
            ]) if len(x) > 0 else 0
        )
        trends['daily_sentiment'] = daily_sentiment.to_dict()
        
        # Hourly pattern
        hourly_sentiment = feedback_df.groupby('hour').apply(
            lambda x: np.mean([
                self.analyze_sentiment(text)['polarity'] 
                for text in x['feedback_text'] if text
            ]) if len(x) > 0 else 0
        )
        trends['hourly_pattern'] = hourly_sentiment.to_dict()
        
        # Most common themes
        all_text = ' '.join(feedback_df['feedback_text'].dropna().astype(str))
        words = self.preprocess_text(all_text).split()
        
        # Filter out common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        word_freq = Counter(filtered_words)
        trends['common_themes'] = dict(word_freq.most_common(10))
        
        return trends
    
    def generate_insights(self, feedback_data, airport_code='DEL'):
        """Generate actionable insights from feedback analysis"""
        insights = {
            'summary': {},
            'recommendations': [],
            'priority_areas': [],
            'sentiment_distribution': {}
        }
        
        if not feedback_data:
            return insights
        
        # Analyze all feedback
        sentiment_scores = []
        aspect_analysis = {}
        
        for feedback in feedback_data:
            if isinstance(feedback, dict) and 'feedback_text' in feedback:
                text = feedback['feedback_text']
            else:
                text = str(feedback)
            
            # Overall sentiment
            sentiment = self.analyze_sentiment(text)
            sentiment_scores.append(sentiment['polarity'])
            
            # Aspect analysis
            aspects = self.analyze_aspects(text)
            for aspect, data in aspects.items():
                if aspect not in aspect_analysis:
                    aspect_analysis[aspect] = []
                aspect_analysis[aspect].append(data['sentiment'])
        
        # Summary statistics
        if sentiment_scores:
            insights['summary'] = {
                'average_sentiment': round(np.mean(sentiment_scores), 3),
                'sentiment_std': round(np.std(sentiment_scores), 3),
                'total_feedback': len(sentiment_scores)
            }
            
            # Sentiment distribution
            positive_count = sum(1 for s in sentiment_scores if s > 0.1)
            negative_count = sum(1 for s in sentiment_scores if s < -0.1)
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            insights['sentiment_distribution'] = {
                'positive': positive_count,
                'neutral': neutral_count,
                'negative': negative_count
            }
        
        # Aspect insights
        priority_areas = []
        for aspect, scores in aspect_analysis.items():
            if scores:
                avg_score = np.mean(scores)
                if avg_score < -0.2:  # Negative aspect
                    priority_areas.append({
                        'aspect': aspect,
                        'score': round(avg_score, 3),
                        'mentions': len(scores)
                    })
        
        # Sort by severity
        priority_areas.sort(key=lambda x: x['score'])
        insights['priority_areas'] = priority_areas[:5]
        
        # Generate recommendations
        recommendations = []
        for area in priority_areas[:3]:
            if area['aspect'] == 'service':
                recommendations.append("Improve staff training and customer service protocols")
            elif area['aspect'] == 'shopping':
                recommendations.append("Review retail pricing and product selection")
            elif area['aspect'] == 'food':
                recommendations.append("Enhance F&B quality and variety")
            elif area['aspect'] == 'facility':
                recommendations.append("Upgrade facility cleanliness and amenities")
            elif area['aspect'] == 'navigation':
                recommendations.append("Improve signage and wayfinding systems")
            elif area['aspect'] == 'wait_time':
                recommendations.append("Optimize queue management and processing times")
        
        insights['recommendations'] = recommendations
        
        return insights
    
    def predict_nps_trend(self, current_nps, sentiment_trend, days_ahead=30):
        """Predict NPS trend based on sentiment analysis"""
        if not sentiment_trend:
            return []
        
        # Simple linear prediction based on recent sentiment trend
        recent_sentiment = list(sentiment_trend.values())[-7:]  # Last 7 days
        if len(recent_sentiment) < 2:
            return []
        
        # Calculate trend slope
        x = np.arange(len(recent_sentiment))
        slope = np.polyfit(x, recent_sentiment, 1)[0]
        
        # Project NPS changes
        predictions = []
        for day in range(1, days_ahead + 1):
            # Convert sentiment change to NPS change
            sentiment_change = slope * day
            nps_change = sentiment_change * 10  # Approximate conversion
            
            predicted_nps = current_nps + nps_change
            predicted_nps = np.clip(predicted_nps, -100, 100)  # NPS bounds
            
            predictions.append({
                'day': day,
                'predicted_nps': round(predicted_nps, 1),
                'confidence': max(0.1, 1 - (day / days_ahead) * 0.7)  # Decreasing confidence
            })
        
        return predictions
    
    def save_model(self, filepath='models/sentiment_model.pkl'):
        """Save model parameters and performance history"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                'sentiment_threshold_promoter': self.sentiment_threshold_promoter,
                'sentiment_threshold_detractor': self.sentiment_threshold_detractor,
                'aspect_keywords': self.aspect_keywords,
                'aspect_weights': self.aspect_weights,
                'performance_history': self.performance_history[-1000:]  # Keep last 1000 records
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            print(f"Error saving sentiment model: {e}")
    
    def load_model(self, filepath='models/sentiment_model.pkl'):
        """Load saved model parameters"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.sentiment_threshold_promoter = model_data.get('sentiment_threshold_promoter', 0.3)
            self.sentiment_threshold_detractor = model_data.get('sentiment_threshold_detractor', -0.1)
            self.aspect_keywords = model_data.get('aspect_keywords', self.aspect_keywords)
            self.aspect_weights = model_data.get('aspect_weights', self.aspect_weights)
            self.performance_history = model_data.get('performance_history', [])
            
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
