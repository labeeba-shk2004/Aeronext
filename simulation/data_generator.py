"""
Data generator for the AeroNexus AI platform.
Creates realistic simulation data for passengers, transactions, feedback, and products.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

class AirportDataGenerator:
    def __init__(self, seed=42):
        """Initialize data generator with random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        self.current_date = datetime.now()
        
        # Data generation parameters
        self.del_daily_passengers = 192000  # 70M / 365
        self.jai_daily_passengers = 18600   # 6.8M / 365
        
        # Passenger segments
        self.del_segments = {'business': 0.55, 'leisure': 0.45}
        self.jai_segments = {'business': 0.30, 'leisure': 0.70}
        
        # Age group distributions
        self.age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        self.age_weights = [0.15, 0.25, 0.25, 0.20, 0.10, 0.05]
        
        # Nationality distribution
        self.nationalities = ['Indian', 'Foreign']
        self.del_nationality_weights = [0.70, 0.30]
        self.jai_nationality_weights = [0.85, 0.15]
        
        # Flight type distribution
        self.flight_types = ['domestic', 'international']
        self.del_flight_weights = [0.60, 0.40]
        self.jai_flight_weights = [0.80, 0.20]
    
    def generate_passengers(self, airport_code, num_days=30, num_passengers_per_day=None):
        """Generate passenger data for specified airport and time period"""
        
        if airport_code == 'DEL':
            daily_passengers = num_passengers_per_day or min(1000, self.del_daily_passengers // 100)  # Reduced for simulation
            segments = self.del_segments
            nationality_weights = self.del_nationality_weights
            flight_weights = self.del_flight_weights
            terminals = ['T3'] * 8 + ['T1', 'T2']  # T3 is primary
            gate_areas = ['A', 'B', 'C', 'D', 'E', 'F']
        else:  # JAI
            daily_passengers = num_passengers_per_day or min(200, self.jai_daily_passengers // 200)  # Reduced for simulation
            segments = self.jai_segments
            nationality_weights = self.jai_nationality_weights
            flight_weights = self.jai_flight_weights
            terminals = ['Domestic', 'International']
            gate_areas = ['A', 'B', 'C']
        
        passengers = []
        
        for day in range(num_days):
            current_day = self.current_date - timedelta(days=num_days - day - 1)
            
            # Vary daily volume (weekends +20%, weekdays normal)
            if current_day.weekday() >= 5:  # Weekend
                day_volume = int(daily_passengers * 1.2)
            else:
                day_volume = daily_passengers
            
            for passenger_num in range(day_volume):
                passenger_id = f"{airport_code}_{day:03d}_{passenger_num:04d}"
                
                # Generate passenger characteristics
                segment = np.random.choice(list(segments.keys()), p=list(segments.values()))
                age_group = np.random.choice(self.age_groups, p=self.age_weights)
                nationality = np.random.choice(self.nationalities, p=nationality_weights)
                flight_type = np.random.choice(self.flight_types, p=flight_weights)
                
                # Generate arrival and departure times
                arrival_hour = max(0, min(23, int(np.random.normal(12, 4))))  # Peak around noon
                arrival_time = current_day.replace(
                    hour=arrival_hour,
                    minute=np.random.randint(0, 60),
                    second=0,
                    microsecond=0
                )
                
                # Dwell time based on flight type and segment
                if flight_type == 'international':
                    base_dwell = 180 if airport_code == 'DEL' else 150
                else:
                    base_dwell = 120 if airport_code == 'DEL' else 90
                
                # Business travelers arrive closer to departure
                if segment == 'business':
                    dwell_multiplier = 0.8
                else:
                    dwell_multiplier = 1.2
                
                dwell_time = max(60, int(np.random.normal(base_dwell * dwell_multiplier, base_dwell * 0.2)))
                departure_time = arrival_time + timedelta(minutes=dwell_time)
                
                # Terminal and gate assignment
                terminal = np.random.choice(terminals)
                gate_area = np.random.choice(gate_areas)
                
                # Generate spending pattern
                base_spend = self._calculate_base_spend(segment, flight_type, nationality, airport_code)
                actual_spend = max(0, np.random.gamma(2, base_spend / 2))
                
                # Items purchased (based on spend)
                items_purchased = max(0, int(actual_spend / 200) + np.random.poisson(1))
                
                # Technology adoption
                app_user = np.random.random() < (0.35 if segment == 'business' else 0.25)
                loyalty_member = np.random.random() < (0.60 if segment == 'business' else 0.30)
                
                # Satisfaction score (influenced by segment and spend)
                base_satisfaction = 3.8 if segment == 'business' else 3.5
                spend_bonus = min(0.5, actual_spend / 1000)
                satisfaction_score = np.clip(
                    np.random.normal(base_satisfaction + spend_bonus, 0.3),
                    1.0, 5.0
                )
                
                passengers.append({
                    'passenger_id': passenger_id,
                    'segment': segment,
                    'age_group': age_group,
                    'nationality': nationality,
                    'flight_type': flight_type,
                    'arrival_time': arrival_time,
                    'departure_time': departure_time,
                    'dwell_time_minutes': dwell_time,
                    'terminal': terminal,
                    'gate_area': gate_area,
                    'total_spend': round(actual_spend, 2),
                    'items_purchased': items_purchased,
                    'app_user': app_user,
                    'loyalty_member': loyalty_member,
                    'satisfaction_score': round(satisfaction_score, 1)
                })
        
        return pd.DataFrame(passengers)
    
    def _calculate_base_spend(self, segment, flight_type, nationality, airport_code):
        """Calculate base spending amount based on passenger characteristics"""
        
        if airport_code == 'DEL':
            base = 400
        else:
            base = 300
        
        # Segment multiplier
        if segment == 'business':
            base *= 1.5
        
        # Flight type multiplier
        if flight_type == 'international':
            base *= 1.3
        
        # Nationality multiplier
        if nationality == 'Foreign':
            base *= 1.4
        
        return base
    
    def generate_products(self):
        """Generate product catalog for both airports"""
        
        products = []
        
        # DEL products
        del_products = [
            # Food & Beverage
            {'name': 'Masala Chai', 'category': 'Food & Beverage', 'base_price': 125.0, 'outlet': 'Café Coffee Day'},
            {'name': 'Butter Chicken Meal', 'category': 'Food & Beverage', 'base_price': 450.0, 'outlet': 'Punjab Grill'},
            {'name': 'Coffee Blend', 'category': 'Food & Beverage', 'base_price': 180.0, 'outlet': 'Starbucks'},
            {'name': 'Fresh Juice', 'category': 'Food & Beverage', 'base_price': 220.0, 'outlet': 'Juice Bar'},
            {'name': 'Sandwich Combo', 'category': 'Food & Beverage', 'base_price': 320.0, 'outlet': 'Subway'},
            
            # Electronics & Gadgets
            {'name': 'Smartphone Charger', 'category': 'Electronics & Gadgets', 'base_price': 800.0, 'outlet': 'Croma'},
            {'name': 'Wireless Headphones', 'category': 'Electronics & Gadgets', 'base_price': 2500.0, 'outlet': 'Croma'},
            {'name': 'Power Bank', 'category': 'Electronics & Gadgets', 'base_price': 1200.0, 'outlet': 'Electronics Store'},
            {'name': 'Travel Adapter', 'category': 'Electronics & Gadgets', 'base_price': 450.0, 'outlet': 'Tech Zone'},
            
            # Fashion & Accessories
            {'name': 'Designer Scarf', 'category': 'Fashion & Accessories', 'base_price': 1200.0, 'outlet': 'Shoppers Stop'},
            {'name': 'Leather Wallet', 'category': 'Fashion & Accessories', 'base_price': 1800.0, 'outlet': 'Hidesign'},
            {'name': 'Sunglasses', 'category': 'Fashion & Accessories', 'base_price': 2200.0, 'outlet': 'Titan Eye+'},
            {'name': 'Watch', 'category': 'Fashion & Accessories', 'base_price': 3500.0, 'outlet': 'Titan'},
            
            # Souvenirs & Gifts
            {'name': 'India Gate Model', 'category': 'Souvenirs & Gifts', 'base_price': 650.0, 'outlet': 'Gift Shop'},
            {'name': 'Handicraft Set', 'category': 'Souvenirs & Gifts', 'base_price': 890.0, 'outlet': 'Craft Corner'},
            {'name': 'Tea Gift Box', 'category': 'Souvenirs & Gifts', 'base_price': 750.0, 'outlet': 'Indian Flavors'},
            
            # Books & Magazines
            {'name': 'Travel Guide Book', 'category': 'Books & Magazines', 'base_price': 450.0, 'outlet': 'WHSmith'},
            {'name': 'Business Magazine', 'category': 'Books & Magazines', 'base_price': 180.0, 'outlet': 'News Stand'},
            {'name': 'Novel', 'category': 'Books & Magazines', 'base_price': 320.0, 'outlet': 'Bookstore'}
        ]
        
        # JAI products  
        jai_products = [
            # Food & Beverage (Higher proportion for JAI)
            {'name': 'Rajasthani Thali', 'category': 'Food & Beverage', 'base_price': 320.0, 'outlet': 'Rajasthani Rasoi'},
            {'name': 'Dal Baati Churma', 'category': 'Food & Beverage', 'base_price': 280.0, 'outlet': 'Local Flavors'},
            {'name': 'Lassi', 'category': 'Food & Beverage', 'base_price': 120.0, 'outlet': 'Dairy Delights'},
            {'name': 'Herbal Tea', 'category': 'Food & Beverage', 'base_price': 95.0, 'outlet': 'Tea Junction'},
            {'name': 'Snack Combo', 'category': 'Food & Beverage', 'base_price': 180.0, 'outlet': 'Quick Bites'},
            
            # Souvenirs & Gifts (Higher proportion for JAI)
            {'name': 'Rajasthani Handicraft', 'category': 'Souvenirs & Gifts', 'base_price': 650.0, 'outlet': 'Jaipur Crafts'},
            {'name': 'Blue Pottery', 'category': 'Souvenirs & Gifts', 'base_price': 450.0, 'outlet': 'Art Gallery'},
            {'name': 'Gemstone Jewelry', 'category': 'Souvenirs & Gifts', 'base_price': 1200.0, 'outlet': 'Gem Palace'},
            {'name': 'Textile Bag', 'category': 'Souvenirs & Gifts', 'base_price': 380.0, 'outlet': 'Craft Store'},
            {'name': 'Miniature Painting', 'category': 'Souvenirs & Gifts', 'base_price': 850.0, 'outlet': 'Art Corner'},
            
            # Fashion & Accessories
            {'name': 'Silver Jewelry', 'category': 'Fashion & Accessories', 'base_price': 2500.0, 'outlet': 'Silver House'},
            {'name': 'Printed Scarf', 'category': 'Fashion & Accessories', 'base_price': 680.0, 'outlet': 'Fashion Point'},
            
            # Electronics & Gadgets
            {'name': 'Travel Pillow', 'category': 'Electronics & Gadgets', 'base_price': 450.0, 'outlet': 'Travel Essentials'},
            {'name': 'Phone Case', 'category': 'Electronics & Gadgets', 'base_price': 280.0, 'outlet': 'Mobile Store'},
            
            # Books & Magazines
            {'name': 'Rajasthan Guide', 'category': 'Books & Magazines', 'base_price': 350.0, 'outlet': 'Book Corner'},
            {'name': 'Local Magazine', 'category': 'Books & Magazines', 'base_price': 80.0, 'outlet': 'News Stand'}
        ]
        
        # Add DEL products
        for i, product in enumerate(del_products, 1):
            products.append({
                'product_id': f'PROD_{i:03d}',
                'name': product['name'],
                'category': product['category'],
                'base_price': product['base_price'],
                'airport_code': 'DEL',
                'outlet_name': product['outlet'],
                'description': f"{product['name']} available at {product['outlet']}",
                'margin_percent': self._get_category_margin(product['category']),
                'stock_level': np.random.randint(20, 200),
                'popularity_score': round(np.random.uniform(6.0, 9.5), 1)
            })
        
        # Add JAI products
        for i, product in enumerate(jai_products, len(del_products) + 1):
            products.append({
                'product_id': f'PROD_{i:03d}',
                'name': product['name'],
                'category': product['category'],
                'base_price': product['base_price'],
                'airport_code': 'JAI',
                'outlet_name': product['outlet'],
                'description': f"{product['name']} available at {product['outlet']}",
                'margin_percent': self._get_category_margin(product['category']),
                'stock_level': np.random.randint(15, 100),
                'popularity_score': round(np.random.uniform(5.5, 9.0), 1)
            })
        
        return pd.DataFrame(products)
    
    def _get_category_margin(self, category):
        """Get typical margin percentage for product category"""
        margins = {
            'Food & Beverage': random.randint(60, 75),
            'Electronics & Gadgets': random.randint(40, 55),
            'Fashion & Accessories': random.randint(70, 90),
            'Souvenirs & Gifts': random.randint(80, 95),
            'Books & Magazines': random.randint(35, 50)
        }
        return margins.get(category, 60)
    
    def generate_transactions(self, passengers_df, products_df, transaction_probability=0.7):
        """Generate transaction data based on passenger and product data"""
        
        transactions = []
        transaction_id = 1
        
        for _, passenger in passengers_df.iterrows():
            if passenger['items_purchased'] == 0:
                continue
                
            # Filter products by airport
            airport_products = products_df[products_df['airport_code'] == passenger['passenger_id'][:3]]
            
            if airport_products.empty:
                continue
            
            # Generate transactions for this passenger
            num_transactions = min(passenger['items_purchased'], np.random.poisson(2) + 1)
            remaining_budget = passenger['total_spend']
            
            for trans_num in range(num_transactions):
                if remaining_budget <= 0:
                    break
                
                # Select product based on passenger segment and category preferences
                selected_product = self._select_product_for_passenger(
                    passenger, airport_products, remaining_budget
                )
                
                if selected_product is None:
                    continue
                
                # Determine quantity (usually 1, sometimes 2-3 for cheaper items)
                if selected_product['base_price'] < 200:
                    quantity = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                else:
                    quantity = 1
                
                # Calculate price with potential discounts
                unit_price = selected_product['base_price']
                discount = 0
                
                # Apply discounts based on conditions
                if passenger['loyalty_member'] and np.random.random() < 0.3:
                    discount = unit_price * 0.1  # 10% loyalty discount
                elif passenger['app_user'] and np.random.random() < 0.2:
                    discount = unit_price * 0.05  # 5% app user discount
                
                total_amount = (unit_price * quantity) - discount
                
                if total_amount > remaining_budget:
                    # Adjust quantity to fit budget
                    quantity = max(1, int(remaining_budget / unit_price))
                    total_amount = (unit_price * quantity) - min(discount, unit_price * quantity * 0.1)
                
                # Generate transaction timestamp (during dwell time)
                dwell_start = passenger['arrival_time']
                dwell_duration = passenger['dwell_time_minutes']
                
                # Transactions more likely in first 70% of dwell time
                transaction_offset = int(np.random.beta(2, 5) * dwell_duration * 0.7)
                transaction_time = dwell_start + timedelta(minutes=transaction_offset)
                
                # Payment method based on passenger characteristics
                if passenger['age_group'] in ['18-24', '25-34'] or passenger['app_user']:
                    payment_method = np.random.choice(['Card', 'UPI', 'Cash'], p=[0.5, 0.4, 0.1])
                else:
                    payment_method = np.random.choice(['Card', 'Cash', 'UPI'], p=[0.6, 0.3, 0.1])
                
                # Outlet location based on terminal and category
                outlet_location = self._get_outlet_location(
                    passenger['terminal'], selected_product['category']
                )
                
                transactions.append({
                    'transaction_id': f'TXN_{transaction_id:06d}',
                    'passenger_id': passenger['passenger_id'],
                    'airport_code': passenger['passenger_id'][:3],
                    'product_id': selected_product['product_id'],
                    'category': selected_product['category'],
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'total_amount': round(total_amount, 2),
                    'timestamp': transaction_time,
                    'payment_method': payment_method,
                    'discount_applied': round(discount, 2),
                    'outlet_location': outlet_location
                })
                
                remaining_budget -= total_amount
                transaction_id += 1
        
        return pd.DataFrame(transactions)
    
    def _select_product_for_passenger(self, passenger, products_df, budget):
        """Select appropriate product for passenger based on preferences and budget"""
        
        # Filter products within budget
        affordable_products = products_df[products_df['base_price'] <= budget]
        
        if affordable_products.empty:
            return None
        
        # Category preferences based on passenger segment
        if passenger['segment'] == 'business':
            category_weights = {
                'Food & Beverage': 0.35,
                'Electronics & Gadgets': 0.25,
                'Fashion & Accessories': 0.20,
                'Books & Magazines': 0.15,
                'Souvenirs & Gifts': 0.05
            }
        else:  # leisure
            category_weights = {
                'Food & Beverage': 0.30,
                'Souvenirs & Gifts': 0.30,
                'Fashion & Accessories': 0.20,
                'Electronics & Gadgets': 0.15,
                'Books & Magazines': 0.05
            }
        
        # Weight products by category preference and popularity
        product_weights = []
        for _, product in affordable_products.iterrows():
            category_weight = category_weights.get(product['category'], 0.1)
            popularity_weight = product['popularity_score'] / 10.0
            final_weight = category_weight * popularity_weight
            product_weights.append(final_weight)
        
        if sum(product_weights) == 0:
            return affordable_products.iloc[0]
        
        # Normalize weights
        product_weights = np.array(product_weights) / sum(product_weights)
        
        # Select product
        selected_idx = np.random.choice(len(affordable_products), p=product_weights)
        return affordable_products.iloc[selected_idx]
    
    def _get_outlet_location(self, terminal, category):
        """Get outlet location based on terminal and product category"""
        
        if terminal == 'T3':
            locations = ['T3-Central Plaza', 'T3-Departures', 'T3-Arrivals', 'T3-Gate Lounge']
        elif terminal in ['T1', 'T2']:
            locations = [f'{terminal}-Departures', f'{terminal}-Central', f'{terminal}-Food Court']
        elif terminal == 'Domestic':
            locations = ['Domestic-Departures', 'Domestic-Arrivals', 'Domestic-Food Court']
        elif terminal == 'International':
            locations = ['International-Departures', 'International-Duty Free', 'International-Lounge']
        else:
            locations = ['Central Plaza', 'Food Court', 'Shopping Area']
        
        # Some categories have preferred locations
        if category == 'Food & Beverage':
            food_locations = [loc for loc in locations if 'Food' in loc or 'Central' in loc]
            if food_locations:
                locations = food_locations
        
        return np.random.choice(locations)

def generate_all_data():
    """Generate all required data files for the simulation"""
    
    print("Generating simulation data...")
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Initialize generator
    generator = AirportDataGenerator(seed=42)
    
    # Generate products first (needed for transactions)
    print("Generating products...")
    products_df = generator.generate_products()
    products_df.to_csv('data/products.csv', index=False)
    print(f"Generated {len(products_df)} products")
    
    # Generate passengers for both airports
    print("Generating passengers...")
    
    # DEL passengers (reduced for simulation)
    del_passengers = generator.generate_passengers('DEL', num_days=30, num_passengers_per_day=100)
    del_passengers.to_csv('data/delhi_passengers.csv', index=False)
    print(f"Generated {len(del_passengers)} DEL passengers")
    
    # JAI passengers (reduced for simulation)
    jai_passengers = generator.generate_passengers('JAI', num_days=30, num_passengers_per_day=50)
    jai_passengers.to_csv('data/jaipur_passengers.csv', index=False)
    print(f"Generated {len(jai_passengers)} JAI passengers")
    
    # Combine passengers for transaction generation
    all_passengers = pd.concat([del_passengers, jai_passengers], ignore_index=True)
    
    # Generate transactions
    print("Generating transactions...")
    transactions_df = generator.generate_transactions(all_passengers, products_df)
    transactions_df.to_csv('data/transactions.csv', index=False)
    print(f"Generated {len(transactions_df)} transactions")
    
    # Generate feedback using the feedback simulator
    print("Generating feedback...")
    from simulation.feedback_simulator import FeedbackSimulator
    
    feedback_sim = FeedbackSimulator()
    
    # Generate feedback for both airports
    del_feedback = []
    jai_feedback = []
    
    for _ in range(50):  # 50 feedback entries per airport
        del_feedback.append(feedback_sim.generate_feedback('DEL', 'mixed'))
        jai_feedback.append(feedback_sim.generate_feedback('JAI', 'mixed'))
    
    all_feedback = del_feedback + jai_feedback
    feedback_df = pd.DataFrame(all_feedback)
    feedback_df.to_csv('data/feedback.csv', index=False)
    print(f"Generated {len(feedback_df)} feedback entries")
    
    print("Data generation completed successfully!")
    return True

if __name__ == "__main__":
    generate_all_data()
