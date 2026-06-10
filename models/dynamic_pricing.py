"""
Q-Learning based dynamic pricing engine for airport retail optimization.
Adjusts prices based on demand, time of day, crowd levels, and historical performance.
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import json

class DynamicPricingAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # exploration rate
        
        # State space: [demand_level, time_period, crowd_level, category]
        self.demand_levels = ['low', 'medium', 'high']
        self.time_periods = ['morning', 'afternoon', 'evening', 'night']
        self.crowd_levels = ['low', 'medium', 'high']
        self.categories = ['Food & Beverage', 'Electronics & Gadgets', 'Fashion & Accessories', 'Souvenirs & Gifts', 'Books & Magazines']
        
        # Action space: price adjustments [-20%, -10%, 0%, +10%, +20%]
        self.price_adjustments = [-0.20, -0.10, 0.0, 0.10, 0.20]
        
        # Initialize Q-table
        self.q_table = {}
        self._initialize_q_table()
        
        # Performance tracking
        self.performance_history = []
        
    def _initialize_q_table(self):
        """Initialize Q-table with random values"""
        for demand in self.demand_levels:
            for time in self.time_periods:
                for crowd in self.crowd_levels:
                    for category in self.categories:
                        state = (demand, time, crowd, category)
                        self.q_table[state] = np.random.uniform(-1, 1, len(self.price_adjustments))
    
    def get_state(self, demand_level, time_of_day, crowd_level, category):
        """Convert current conditions to state representation"""
        # Map time to period
        hour = time_of_day if isinstance(time_of_day, int) else datetime.now().hour
        if 6 <= hour < 12:
            time_period = 'morning'
        elif 12 <= hour < 18:
            time_period = 'afternoon'
        elif 18 <= hour <= 23:
            time_period = 'evening'
        else:
            time_period = 'night'
        
        return (demand_level, time_period, crowd_level, category)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(-1, 1, len(self.price_adjustments))
        
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(len(self.price_adjustments))
        else:
            # Exploitation: best known action
            return np.argmax(self.q_table[state])
    
    def get_price_adjustment(self, demand_level, time_of_day, crowd_level, category):
        """Get recommended price adjustment for given conditions"""
        state = self.get_state(demand_level, time_of_day, crowd_level, category)
        action = self.select_action(state)
        adjustment = self.price_adjustments[action]
        
        # Add business logic constraints
        adjustment = self._apply_business_constraints(adjustment, category, crowd_level)
        
        return adjustment, action, state
    
    def _apply_business_constraints(self, adjustment, category, crowd_level):
        """Apply business rules to limit extreme price changes"""
        # Limit price increases during low crowd periods
        if crowd_level == 'low' and adjustment > 0.10:
            adjustment = 0.10
        
        # Be more aggressive with F&B pricing during peak times
        if category == 'Food & Beverage' and crowd_level == 'high':
            adjustment = min(adjustment * 1.2, 0.20)
        
        # Conservative pricing for luxury items
        if category in ['Electronics & Gadgets', 'Fashion & Accessories'] and adjustment > 0.15:
            adjustment = 0.15
        
        return adjustment
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value based on observed reward"""
        if state not in self.q_table:
            self.q_table[state] = np.random.uniform(-1, 1, len(self.price_adjustments))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.random.uniform(-1, 1, len(self.price_adjustments))
        
        # Q-learning update rule
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * next_max - old_value
        )
        
        self.q_table[state][action] = new_value
        
        # Record performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'state': state,
            'action': action,
            'reward': reward,
            'q_value': new_value
        })
    
    def calculate_reward(self, base_price, adjusted_price, units_sold, base_units, profit_margin):
        """Calculate reward for pricing decision"""
        # Revenue impact
        revenue_change = (adjusted_price * units_sold) - (base_price * base_units)
        
        # Volume impact (penalize large volume drops)
        volume_change_pct = (units_sold - base_units) / max(base_units, 1)
        volume_penalty = min(0, volume_change_pct) * 2  # Negative penalty for volume loss
        
        # Profit margin consideration
        profit_change = revenue_change * profit_margin
        
        # Normalize reward to [-1, 1] range
        reward = np.tanh(profit_change / 1000) + volume_penalty * 0.3
        
        return reward
    
    def simulate_pricing_scenario(self, base_price, category, hours=24):
        """Simulate pricing decisions over time period"""
        results = []
        current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for hour in range(hours):
            # Simulate varying conditions throughout the day
            time_of_day = hour
            
            # Model demand patterns
            if 6 <= hour < 10 or 18 <= hour < 22:
                demand_level = np.random.choice(['medium', 'high'], p=[0.3, 0.7])
                crowd_level = np.random.choice(['medium', 'high'], p=[0.4, 0.6])
            elif 10 <= hour < 18:
                demand_level = np.random.choice(['low', 'medium'], p=[0.4, 0.6])
                crowd_level = np.random.choice(['low', 'medium'], p=[0.5, 0.5])
            else:
                demand_level = np.random.choice(['low', 'medium'], p=[0.7, 0.3])
                crowd_level = np.random.choice(['low', 'medium'], p=[0.8, 0.2])
            
            # Get pricing recommendation
            adjustment, action, state = self.get_price_adjustment(
                demand_level, time_of_day, crowd_level, category
            )
            
            adjusted_price = base_price * (1 + adjustment)
            
            # Simulate sales (simplified model)
            base_demand = self._get_base_demand(category, time_of_day)
            price_elasticity = self._get_price_elasticity(category)
            demand_multiplier = self._get_demand_multiplier(demand_level, crowd_level)
            
            # Sales = base_demand * demand_multiplier * price_effect
            price_effect = (1 - adjustment * price_elasticity)
            expected_sales = max(0, base_demand * demand_multiplier * price_effect)
            
            # Add some randomness
            actual_sales = max(0, np.random.poisson(expected_sales))
            
            results.append({
                'hour': hour,
                'time_period': self.get_state(demand_level, time_of_day, crowd_level, category)[1],
                'demand_level': demand_level,
                'crowd_level': crowd_level,
                'base_price': base_price,
                'price_adjustment': adjustment,
                'adjusted_price': adjusted_price,
                'expected_sales': expected_sales,
                'actual_sales': actual_sales,
                'revenue': adjusted_price * actual_sales,
                'state': state,
                'action': action
            })
        
        return pd.DataFrame(results)
    
    def _get_base_demand(self, category, hour):
        """Get base demand for category at given hour"""
        base_demands = {
            'Food & Beverage': 20,
            'Electronics & Gadgets': 5,
            'Fashion & Accessories': 8,
            'Souvenirs & Gifts': 12,
            'Books & Magazines': 6
        }
        
        # Time of day multipliers
        if 6 <= hour < 10:
            multiplier = 1.2  # Morning rush
        elif 12 <= hour < 14:
            multiplier = 1.5  # Lunch time
        elif 18 <= hour < 22:
            multiplier = 1.3  # Evening peak
        else:
            multiplier = 0.6  # Off-peak
        
        return base_demands.get(category, 10) * multiplier
    
    def _get_price_elasticity(self, category):
        """Get price elasticity for category"""
        elasticities = {
            'Food & Beverage': 0.8,      # Less elastic (necessity)
            'Electronics & Gadgets': 1.2,  # More elastic (discretionary)
            'Fashion & Accessories': 1.1,
            'Souvenirs & Gifts': 0.9,
            'Books & Magazines': 1.0
        }
        return elasticities.get(category, 1.0)
    
    def _get_demand_multiplier(self, demand_level, crowd_level):
        """Get demand multiplier based on conditions"""
        demand_multipliers = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.4
        }
        
        crowd_multipliers = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.3
        }
        
        return demand_multipliers[demand_level] * crowd_multipliers[crowd_level]
    
    def calculate_revenue_uplift(self, simulation_results):
        """Calculate revenue uplift from dynamic pricing"""
        if simulation_results.empty:
            return 0.0
        
        # Compare with baseline (no price adjustment)
        baseline_revenue = simulation_results['base_price'].sum() * simulation_results['expected_sales'].mean()
        actual_revenue = simulation_results['revenue'].sum()
        
        uplift = (actual_revenue - baseline_revenue) / max(baseline_revenue, 1) * 100
        return uplift
    
    def get_pricing_insights(self, category, airport_code='DEL'):
        """Get actionable pricing insights"""
        insights = {
            'recommended_adjustments': {},
            'performance_metrics': {},
            'best_practices': []
        }
        
        # Analyze Q-table for best actions in different scenarios
        for state, q_values in self.q_table.items():
            if state[3] == category:  # Filter by category
                best_action = np.argmax(q_values)
                best_adjustment = self.price_adjustments[best_action]
                
                scenario = f"{state[0]}_demand_{state[1]}_{state[2]}_crowd"
                insights['recommended_adjustments'][scenario] = {
                    'adjustment': best_adjustment,
                    'confidence': np.max(q_values)
                }
        
        # Generate best practices
        if category == 'Food & Beverage':
            insights['best_practices'] = [
                "Increase prices by 10-15% during meal times (12-14h, 19-21h)",
                "Offer discounts during off-peak hours to drive volume",
                "Premium pricing during high crowd periods"
            ]
        elif category == 'Electronics & Gadgets':
            insights['best_practices'] = [
                "Conservative pricing increases (max 10%)",
                "Bundle deals during low demand periods",
                "Premium pricing for business travelers"
            ]
        
        return insights
    
    def save_model(self, filepath='models/pricing_model.pkl'):
        """Save the Q-table and model parameters"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            model_data = {
                'q_table': self.q_table,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'performance_history': self.performance_history[-1000:]  # Keep last 1000 records
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            print(f"Error saving pricing model: {e}")
    
    def load_model(self, filepath='models/pricing_model.pkl'):
        """Load saved Q-table and model parameters"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            self.q_table = model_data.get('q_table', {})
            self.learning_rate = model_data.get('learning_rate', self.learning_rate)
            self.discount_factor = model_data.get('discount_factor', self.discount_factor)
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.performance_history = model_data.get('performance_history', [])
            
        except Exception as e:
            print(f"Error loading pricing model: {e}")
            self._initialize_q_table()  # Initialize if loading fails
