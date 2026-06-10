"""
Domain-aware collaborative filtering recommendation engine for personalized retail offers.
Uses sklearn-based matrix factorization with domain filtering (retail, f&b, lounge).
"""

import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.airport_profiles import AIRPORT_PROFILES

class DomainRecommender:
    def __init__(self, no_components: int = 32, learning_rate: float = 0.05, loss: str = 'warp'):
        """Initialize domain-aware recommender system"""
        self.model = NMF(n_components=no_components, random_state=42, max_iter=200)
        self.user_item_matrix = None
        self.products_df = None
        self.scaler = MinMaxScaler()
        self.item_similarity = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.domains = ['retail', 'f&b', 'lounge']
        
    def fit(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame):
        """
        Train the domain-aware recommendation model
        
        Args:
            interactions_df: DataFrame with columns [user_id, item_id, rating]
            products_df: DataFrame with columns [item_id, domain, name, price, category]
        """
        # Store products with domain information
        self.products_df = products_df.copy()
        
        # Ensure domain column exists and normalize values
        if 'domain' not in products_df.columns:
            # Map categories to domains
            domain_mapping = {
                'Food & Beverage': 'f&b',
                'Coffee': 'f&b',
                'Restaurant': 'f&b',
                'Snacks': 'f&b',
                'Retail': 'retail',
                'Electronics': 'retail',
                'Books': 'retail',
                'Souvenirs': 'retail',
                'Lounge': 'lounge',
                'Premium Lounge': 'lounge'
            }
            self.products_df['domain'] = products_df['category'].map(
                lambda x: domain_mapping.get(x, 'retail')
            )
        
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot(
            index='user_id', 
            columns='item_id', 
            values='rating'
        ).fillna(0)
        
        # Create mappings
        self.user_mapping = {user: idx for idx, user in enumerate(self.user_item_matrix.index)}
        self.item_mapping = {item: idx for idx, item in enumerate(self.user_item_matrix.columns)}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Scale the ratings
        user_item_scaled = self.scaler.fit_transform(self.user_item_matrix.values)
        
        # Train NMF model
        self.model.fit(user_item_scaled)
        
        # Calculate item similarity matrix for domain filtering
        item_features = self.model.components_.T
        self.item_similarity = cosine_similarity(item_features)
        
        print(f"Model trained with {len(self.user_item_matrix)} users and {len(self.user_item_matrix.columns)} products")
        return self
    
    def recommend(self, user_id: str, domain: str, n: int = 5) -> List[Dict]:
        """
        Get domain-filtered recommendations for a user based on NMF predictions.
        """
        if self.user_item_matrix is None or self.products_df is None or self.model is None:
            # Fallback if model not trained or data not loaded
            return self._get_fallback_recommendations(domain, n)

        # Handle new users (cold start)
        if user_id not in self.user_mapping:
            # For new users, use a simpler popularity-based or domain-specific fallback
            # For this demo, we'll use the existing _get_fallback_recommendations
            return self._get_fallback_recommendations(domain, n)

        user_idx = self.user_mapping[user_id]
        
        # Get user latent features
        user_features = self.model.transform(self.user_item_matrix.iloc[user_idx].values.reshape(1, -1))
        
        # Predict ratings for all items
        predicted_ratings_scaled = self.model.inverse_transform(user_features)
        
        # Inverse transform to original rating scale
        predicted_ratings = self.scaler.inverse_transform(predicted_ratings_scaled).flatten()
        
        # Create a series of predicted ratings for all items, mapping to item_ids
        all_item_predictions = pd.Series(predicted_ratings, index=self.user_item_matrix.columns)
        
        # Filter products by domain
        domain_products_df = self.products_df[self.products_df['domain'].str.lower() == domain.lower()]
        
        # Get predicted ratings only for items in the selected domain and that exist in item_mapping
        relevant_predictions = all_item_predictions[all_item_predictions.index.isin(domain_products_df['item_id'])]
        
        # Sort recommendations by predicted rating in descending order
        # Exclude items the user has already interacted with (rated > 0 in user_item_matrix)
        user_interacted_items = self.user_item_matrix.iloc[user_idx][self.user_item_matrix.iloc[user_idx] > 0].index
        
        # Remove already interacted items from recommendations
        recommendations_candidates = relevant_predictions[~relevant_predictions.index.isin(user_interacted_items)]
        
        # Get top N items
        top_n_items = recommendations_candidates.nlargest(n).index.tolist()
        
        recommendations_list = []
        for item_id in top_n_items:
            product_info = domain_products_df[domain_products_df['item_id'] == item_id].iloc[0]
            
            predicted_score = recommendations_candidates[item_id]
            
            # Map predicted score to a more human-readable rating (e.g., 1-5) and confidence (0-1)
            # This is a heuristic and might need tuning based on actual data distribution
            display_rating = max(1.0, min(5.0, (predicted_score / self.scaler.data_max_.max()) * 5.0))
            confidence = max(0.1, min(1.0, predicted_score / self.scaler.data_max_.max())) # Simple confidence

            recommendations_list.append({
                'product_id': item_id,
                'predicted_rating': round(display_rating, 1),
                'confidence': round(confidence, 2),
                'product_name': product_info['name'],
                'category': product_info['category'],
                'domain': product_info['domain'],
                'price': product_info['price'],
                'discount': product_info.get('discount', ''),
                'brand': product_info.get('brand', ''),
                'restaurant': product_info.get('restaurant', ''),
                'lounge': product_info.get('lounge', ''),
                'rank': len(recommendations_list) + 1
            })
            
        # If not enough recommendations from NMF, pad with fallback
        if len(recommendations_list) < n:
            fallback_recs = self._get_fallback_recommendations(domain, n - len(recommendations_list))
            existing_ids = {rec['product_id'] for rec in recommendations_list}
            for rec in fallback_recs:
                if rec['product_id'] not in existing_ids:
                    recommendations_list.append(rec)
                    if len(recommendations_list) == n:
                        break
        
        return recommendations_list
    
    def _get_fallback_recommendations(self, domain: str, n: int) -> List[Dict]:
        """Fallback recommendations when model fails"""
        fallback_by_domain = {
            'retail': [
                {'product_id': 'RET001', 'name': 'Duty Free Electronics - 20% OFF', 'price': 2400, 'discount': '20% OFF', 'brand': 'Samsung Galaxy Buds Pro'},
                {'product_id': 'RET002', 'name': 'Rajasthani Handicrafts Store', 'price': 850, 'discount': '15% OFF', 'brand': 'Traditional Artifacts'},
                {'product_id': 'RET003', 'name': 'Luxury Perfumes & Cosmetics', 'price': 3200, 'discount': '25% OFF', 'brand': 'Chanel & Dior'},
                {'product_id': 'RET004', 'name': 'Travel Essentials Store', 'price': 450, 'discount': '10% OFF', 'brand': 'VIP Luggage & Accessories'},
                {'product_id': 'RET005', 'name': 'Indian Textiles & Fabrics', 'price': 1200, 'discount': '30% OFF', 'brand': 'Fabindia Collection'},
                {'product_id': 'RET006', 'name': 'Electronics & Gadgets Hub', 'price': 1800, 'discount': '12% OFF', 'brand': 'Apple & Sony Products'},
                {'product_id': 'RET007', 'name': 'Jewelry & Watches Store', 'price': 5500, 'discount': '18% OFF', 'brand': 'Titan & Tanishq'}
            ],
            'f&b': [
                {'product_id': 'FB001', 'name': 'Starbucks Coffee - DEL Terminal 3', 'price': 280, 'discount': 'Buy 2 Get 1 Free', 'restaurant': 'Starbucks'},
                {'product_id': 'FB002', 'name': 'Punjab Grill Restaurant', 'price': 1200, 'discount': '15% OFF', 'restaurant': 'Punjab Grill'},
                {'product_id': 'FB003', 'name': 'McDonald\'s Value Meal', 'price': 350, 'discount': '20% OFF on Combo', 'restaurant': 'McDonald\'s'},
                {'product_id': 'FB004', 'name': 'Cafe Coffee Day Express', 'price': 180, 'discount': 'Free Cookie with Coffee', 'restaurant': 'CCD'},
                {'product_id': 'FB005', 'name': 'Haldiram\'s Indian Snacks', 'price': 220, 'discount': '10% OFF', 'restaurant': 'Haldiram\'s'},
                {'product_id': 'FB006', 'name': 'Costa Coffee Premium', 'price': 320, 'discount': 'Happy Hour 50% OFF', 'restaurant': 'Costa Coffee'},
                {'product_id': 'FB007', 'name': 'Wow! Momo Food Court', 'price': 280, 'discount': '25% OFF on Orders Above ₹500', 'restaurant': 'Wow! Momo'},
                {'product_id': 'FB008', 'name': 'Burger King Airport Special', 'price': 420, 'discount': 'Free Upgrade to Large Fries', 'restaurant': 'Burger King'}
            ],
            'lounge': [
                {'product_id': 'LNG001', 'name': 'Plaza Premium Lounge Access', 'price': 1800, 'discount': '30% OFF for 3+ Hours', 'lounge': 'Plaza Premium'},
                {'product_id': 'LNG002', 'name': 'Delhi Airport Spa Services', 'price': 2500, 'discount': '20% OFF Massage Packages', 'lounge': 'O2 Spa'},
                {'product_id': 'LNG003', 'name': 'ITC Hotels Lounge - Premium', 'price': 3200, 'discount': 'Complimentary WiFi & Meals', 'lounge': 'ITC Hotels'},
                {'product_id': 'LNG004', 'name': 'Sleep Pods - Snooze at My Space', 'price': 800, 'discount': '15% OFF for 6+ Hours', 'lounge': 'Snooze Pods'},
                {'product_id': 'LNG005', 'name': 'Business Center & Meeting Rooms', 'price': 1200, 'discount': 'Free Printing & Scanning', 'lounge': 'Regus Business Center'},
                {'product_id': 'LNG006', 'name': 'Jaipur Airport VIP Lounge', 'price': 1500, 'discount': '25% OFF Local Cuisine Buffet', 'lounge': 'Royal Rajasthan Lounge'}
            ]
        }
        
        domain_products = fallback_by_domain.get(domain.lower(), fallback_by_domain['retail'])
        recommendations = []
        
        for i, product in enumerate(domain_products[:n]):
            recommendations.append({
                'product_id': product['product_id'],
                'predicted_rating': 4.0 - (i * 0.2),
                'confidence': 0.8 - (i * 0.1),
                'product_name': product['name'],
                'category': domain.title(),
                'domain': domain,
                'price': product['price'],
                'discount': product.get('discount', ''),
                'brand': product.get('brand', ''),
                'restaurant': product.get('restaurant', ''),
                'lounge': product.get('lounge', ''),
                'rank': i + 1
            })
        
        return recommendations
    
    def get_domain_products(self, domain: str) -> pd.DataFrame:
        """Get all products in a specific domain"""
        if self.products_df is not None:
            return self.products_df[
                self.products_df['domain'].str.lower() == domain.lower()
            ].copy()
        return pd.DataFrame()
    
    def calculate_conversion_lift(self, recommendations: List[Dict], baseline_conversion: float = 0.15) -> Dict:
        """Calculate expected conversion lift from recommendations"""
        if not recommendations:
            return {
                'baseline_conversion': baseline_conversion,
                'predicted_conversion': baseline_conversion,
                'conversion_lift': 0.0,
                'expected_revenue_uplift': 0.0,
                'recommendation_count': 0
            }
        
        # Calculate weighted conversion based on confidence scores
        avg_confidence = sum(r['confidence'] for r in recommendations) / len(recommendations)
        predicted_conversion = baseline_conversion * (1 + avg_confidence)
        
        conversion_lift = (predicted_conversion - baseline_conversion) / baseline_conversion * 100
        
        # Calculate expected revenue uplift
        avg_price = sum(r['price'] for r in recommendations) / len(recommendations)
        expected_revenue_uplift = (predicted_conversion - baseline_conversion) * avg_price
        
        return {
            'baseline_conversion': baseline_conversion,
            'predicted_conversion': round(predicted_conversion, 3),
            'conversion_lift': round(conversion_lift, 1),
            'expected_revenue_uplift': round(expected_revenue_uplift, 2),
            'avg_recommendation_price': round(avg_price, 2),
            'recommendation_count': len(recommendations)
        }
    
    def save(self, path: str):
        """Save trained model and dataset"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model_data = {
                'model': self.model,
                'user_item_matrix': self.user_item_matrix,
                'products_df': self.products_df,
                'scaler': self.scaler,
                'item_similarity': self.item_similarity,
                'user_mapping': self.user_mapping,
                'item_mapping': self.item_mapping,
                'reverse_item_mapping': self.reverse_item_mapping
            }
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    @classmethod
    def load(cls, path: str):
        """Load trained model and dataset"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            instance = cls()
            instance.model = model_data['model']
            instance.user_item_matrix = model_data['user_item_matrix']
            instance.products_df = model_data['products_df']
            instance.scaler = model_data['scaler']
            instance.item_similarity = model_data['item_similarity']
            instance.user_mapping = model_data['user_mapping']
            instance.item_mapping = model_data['item_mapping']
            instance.reverse_item_mapping = model_data['reverse_item_mapping']
            
            print(f"Model loaded from {path}")
            return instance
        except Exception as e:
            print(f"Error loading model: {e}")
            return cls()

# Legacy compatibility class
class AirportRecommender(DomainRecommender):
    """Legacy wrapper for backward compatibility"""
    
    def __init__(self):
        super().__init__()
    
    def train_model(self):
        """Legacy method for training"""
        # Generate sample data if no real data exists
        interactions_df = self._generate_sample_interactions()
        products_df = self._generate_sample_products()
        return self.fit(interactions_df, products_df)
    
    def get_user_recommendations(self, user_id: str, airport_code: str, passenger_segment: str, n_recommendations: int = 5):
        """Legacy method for getting recommendations"""
        # Default to retail domain for legacy compatibility
        return self.recommend(user_id, 'retail', n_recommendations)
    
    def _generate_sample_interactions(self):
        """Generate sample interaction data"""
        np.random.seed(42)
        
        passengers = [f'P{i:04d}' for i in range(1, 501)]
        products = [f'PROD_{i:03d}' for i in range(1, 51)]
        
        data = []
        for passenger in passengers:
            n_purchases = np.random.randint(1, 6)
            purchased_products = np.random.choice(products, n_purchases, replace=False)
            
            for product in purchased_products:
                rating = np.random.uniform(1, 5)
                data.append({
                    'user_id': passenger,
                    'item_id': product,
                    'rating': rating
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_products(self):
        """Generate sample product data with domains"""
        np.random.seed(42)
        
        products = [f'PROD_{i:03d}' for i in range(1, 51)]
        categories = ['Food & Beverage', 'Retail', 'Electronics', 'Books', 'Souvenirs', 'Lounge']
        domains = ['f&b', 'retail', 'retail', 'retail', 'retail', 'lounge']
        
        data = []
        for i, product_id in enumerate(products):
            category_idx = i % len(categories)
            data.append({
                'item_id': product_id,
                'name': f'Product {i+1}',
                'category': categories[category_idx],
                'domain': domains[category_idx],
                'price': np.random.uniform(50, 500)
            })
        
        return pd.DataFrame(data)