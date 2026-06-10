"""
Airport configuration profiles for DEL and JAI airports.
Contains baseline metrics, passenger segments, and operational parameters.
"""

AIRPORT_PROFILES = {
    'DEL': {
        'name': 'Delhi Indira Gandhi International Airport',
        'code': 'DEL',
        'passenger_volume': 70000000,  # 70M+ passengers in 2024
        'retail_density': 21,  # shops per 1000 passengers
        'tech_readiness': 'Advanced',
        'business_segment': 0.55,  # 55% business travelers
        'leisure_segment': 0.45,   # 45% leisure travelers
        'baseline_revenue_per_pax': 312,  # ₹312 baseline
        'target_revenue_per_pax': 390,    # ₹390 target
        'baseline_nps': 52,
        'target_nps': 65,
        'baseline_dwell_time': 104,  # minutes
        'dwell_time_variance': 22,   # ±22 minutes
        'target_dwell_variance': 15, # ±15 minutes target
        'commercial_zones': [
            'Terminal 3 - Departures Level 3',
            'Terminal 3 - Arrivals Level 1', 
            'Terminal 3 - Central Plaza',
            'Terminal 3 - Gate Lounges'
        ],
        'retail_categories': {
            'Fashion & Accessories': 0.25,
            'Electronics & Gadgets': 0.20,
            'Food & Beverage': 0.30,
            'Books & Magazines': 0.10,
            'Souvenirs & Gifts': 0.15
        },
        'peak_hours': {
            'morning': (6, 10),
            'afternoon': (14, 18), 
            'evening': (20, 23)
        },
        'currency': 'INR'
    },
    
    'JAI': {
        'name': 'Jaipur International Airport',
        'code': 'JAI',
        'passenger_volume': 6800000,   # 6.8M+ passengers in 2024
        'retail_density': 9,           # shops per 1000 passengers
        'tech_readiness': 'Growing',
        'business_segment': 0.30,      # 30% business travelers
        'leisure_segment': 0.70,       # 70% leisure travelers
        'baseline_revenue_per_pax': 280,  # Lower baseline for tier-2 airport
        'target_revenue_per_pax': 350,    # Proportional target
        'baseline_nps': 48,
        'target_nps': 58,
        'baseline_dwell_time': 85,     # Lower dwell time for regional airport
        'dwell_time_variance': 18,
        'target_dwell_variance': 12,
        'commercial_zones': [
            'Domestic Terminal - Departures',
            'Domestic Terminal - Arrivals',
            'International Terminal - Departures',
            'Central Food Court'
        ],
        'retail_categories': {
            'Fashion & Accessories': 0.20,
            'Electronics & Gadgets': 0.15,
            'Food & Beverage': 0.40,  # Higher F&B focus
            'Books & Magazines': 0.08,
            'Souvenirs & Gifts': 0.17  # Higher souvenirs for tourism
        },
        'peak_hours': {
            'morning': (7, 11),
            'afternoon': (15, 19),
            'evening': (21, 24)
        },
        'currency': 'INR'
    }
}

# KPI Targets based on pilot proposal
KPI_TARGETS = {
    'revenue_uplift_percent': 25,      # 25% uplift target
    'nps_improvement': 10,             # +10 points NPS
    'dwell_time_variance_reduction': 30,  # 30% variance reduction
    'retail_conversion_uplift': 10,    # 10% conversion improvement
    'queue_time_reduction': 20,        # 20% queue time reduction
    'space_utilization_improvement': 10,  # 10% space utilization
    'app_adoption_target': 25          # 25% app adoption rate
}

# Global constants
GLOBAL_CONSTANTS = {
    'pilot_duration_days': 90,
    'sensors_del': 500,     # Bluetooth beacons
    'sensors_jai': 200,     # Fewer sensors for smaller airport
    'thermal_sensors_del': 200,
    'thermal_sensors_jai': 80,
    'data_latency_seconds': 15,
    'staff_training_hours': 40
}
