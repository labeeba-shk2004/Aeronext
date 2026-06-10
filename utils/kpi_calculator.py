"""
KPI Calculator for the AeroNexus AI platform.
Calculates key performance indicators for airport revenue optimization including
revenue uplift, NPS improvements, dwell time optimization, and operational efficiency metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.airport_profiles import AIRPORT_PROFILES, KPI_TARGETS

class KPICalculator:
    def __init__(self):
        """Initialize KPI calculator with baseline metrics"""
        self.airport_profiles = AIRPORT_PROFILES
        self.kpi_targets = KPI_TARGETS
        
    def calculate_revenue_uplift(self, airport_code: str, current_revenue_per_pax: float, 
                               baseline_revenue_per_pax: Optional[float] = None) -> Dict:
        """Calculate revenue uplift metrics"""
        
        if baseline_revenue_per_pax is None:
            baseline_revenue_per_pax = self.airport_profiles[airport_code]['baseline_revenue_per_pax']
        
        target_revenue_per_pax = self.airport_profiles[airport_code]['target_revenue_per_pax']
        
        # Calculate uplift metrics
        absolute_uplift = current_revenue_per_pax - baseline_revenue_per_pax
        percentage_uplift = (absolute_uplift / baseline_revenue_per_pax) * 100 if baseline_revenue_per_pax > 0 else 0
        
        # Progress towards target
        target_uplift = target_revenue_per_pax - baseline_revenue_per_pax
        progress_to_target = (absolute_uplift / target_uplift) * 100 if target_uplift > 0 else 0
        
        # Annual impact calculation
        annual_passengers = self.airport_profiles[airport_code]['passenger_volume']
        annual_impact = absolute_uplift * annual_passengers
        
        return {
            'baseline_revenue_per_pax': baseline_revenue_per_pax,
            'current_revenue_per_pax': current_revenue_per_pax,
            'target_revenue_per_pax': target_revenue_per_pax,
            'absolute_uplift': round(absolute_uplift, 2),
            'percentage_uplift': round(percentage_uplift, 1),
            'progress_to_target': round(min(100, progress_to_target), 1),
            'annual_impact': round(annual_impact, 0),
            'annual_impact_cr': round(annual_impact / 10000000, 1),  # Convert to Crores
            'target_achieved': current_revenue_per_pax >= target_revenue_per_pax
        }
    
    def calculate_nps_metrics(self, airport_code: str, current_nps: float,
                             promoters: int = 0, passives: int = 0, detractors: int = 0) -> Dict:
        """Calculate NPS-related metrics"""
        
        baseline_nps = self.airport_profiles[airport_code]['baseline_nps']
        target_nps = self.airport_profiles[airport_code]['target_nps']
        
        # NPS improvement
        nps_improvement = current_nps - baseline_nps
        target_improvement = target_nps - baseline_nps
        progress_to_target = (nps_improvement / target_improvement) * 100 if target_improvement > 0 else 0
        
        # Distribution calculations
        total_responses = promoters + passives + detractors
        if total_responses > 0:
            promoter_percentage = (promoters / total_responses) * 100
            passive_percentage = (passives / total_responses) * 100
            detractor_percentage = (detractors / total_responses) * 100
            
            # Verify NPS calculation
            calculated_nps = promoter_percentage - detractor_percentage
        else:
            promoter_percentage = passive_percentage = detractor_percentage = 0
            calculated_nps = current_nps
        
        # NPS category assessment
        if current_nps >= 50:
            nps_category = 'Excellent'
        elif current_nps >= 0:
            nps_category = 'Good'
        elif current_nps >= -50:
            nps_category = 'Poor'
        else:
            nps_category = 'Critical'
        
        return {
            'baseline_nps': baseline_nps,
            'current_nps': current_nps,
            'target_nps': target_nps,
            'nps_improvement': round(nps_improvement, 1),
            'progress_to_target': round(min(100, progress_to_target), 1),
            'promoter_percentage': round(promoter_percentage, 1),
            'passive_percentage': round(passive_percentage, 1),
            'detractor_percentage': round(detractor_percentage, 1),
            'total_responses': total_responses,
            'nps_category': nps_category,
            'target_achieved': current_nps >= target_nps,
            'calculated_nps': round(calculated_nps, 1)
        }
    
    def calculate_dwell_time_metrics(self, airport_code: str, current_avg_dwell: float,
                                   current_variance: float) -> Dict:
        """Calculate dwell time optimization metrics"""
        
        baseline_dwell = self.airport_profiles[airport_code]['baseline_dwell_time']
        baseline_variance = self.airport_profiles[airport_code]['dwell_time_variance']
        target_variance = self.airport_profiles[airport_code]['target_dwell_variance']
        
        # Dwell time changes
        dwell_time_change = current_avg_dwell - baseline_dwell
        dwell_time_change_pct = (dwell_time_change / baseline_dwell) * 100 if baseline_dwell > 0 else 0
        
        # Variance reduction
        variance_reduction = baseline_variance - current_variance
        variance_reduction_pct = (variance_reduction / baseline_variance) * 100 if baseline_variance > 0 else 0
        
        # Progress towards target variance
        target_reduction = baseline_variance - target_variance
        progress_to_target = (variance_reduction / target_reduction) * 100 if target_reduction > 0 else 0
        
        # Efficiency assessment
        if current_variance <= target_variance:
            efficiency_status = 'Optimized'
        elif variance_reduction_pct >= 50:
            efficiency_status = 'Improving'
        elif variance_reduction_pct > 0:
            efficiency_status = 'Stable'
        else:
            efficiency_status = 'Needs Attention'
        
        return {
            'baseline_dwell_time': baseline_dwell,
            'current_dwell_time': current_avg_dwell,
            'dwell_time_change': round(dwell_time_change, 1),
            'dwell_time_change_pct': round(dwell_time_change_pct, 1),
            'baseline_variance': baseline_variance,
            'current_variance': current_variance,
            'target_variance': target_variance,
            'variance_reduction': round(variance_reduction, 1),
            'variance_reduction_pct': round(variance_reduction_pct, 1),
            'progress_to_target': round(min(100, progress_to_target), 1),
            'efficiency_status': efficiency_status,
            'target_achieved': current_variance <= target_variance
        }
    
    def calculate_conversion_metrics(self, total_passengers: int, purchasing_passengers: int,
                                   total_revenue: float, baseline_conversion_rate: float = 0.15) -> Dict:
        """Calculate retail conversion and engagement metrics"""
        
        # Conversion rate calculation
        current_conversion_rate = purchasing_passengers / total_passengers if total_passengers > 0 else 0
        conversion_uplift = current_conversion_rate - baseline_conversion_rate
        conversion_uplift_pct = (conversion_uplift / baseline_conversion_rate) * 100 if baseline_conversion_rate > 0 else 0
        
        # Revenue per visitor
        revenue_per_visitor = total_revenue / total_passengers if total_passengers > 0 else 0
        revenue_per_purchaser = total_revenue / purchasing_passengers if purchasing_passengers > 0 else 0
        
        # Target achievement
        target_conversion_rate = baseline_conversion_rate * (1 + self.kpi_targets['retail_conversion_uplift'] / 100)
        target_achieved = current_conversion_rate >= target_conversion_rate
        
        return {
            'total_passengers': total_passengers,
            'purchasing_passengers': purchasing_passengers,
            'baseline_conversion_rate': round(baseline_conversion_rate * 100, 1),
            'current_conversion_rate': round(current_conversion_rate * 100, 1),
            'target_conversion_rate': round(target_conversion_rate * 100, 1),
            'conversion_uplift_pct': round(conversion_uplift_pct, 1),
            'revenue_per_visitor': round(revenue_per_visitor, 2),
            'revenue_per_purchaser': round(revenue_per_purchaser, 2),
            'total_revenue': round(total_revenue, 2),
            'target_achieved': target_achieved
        }
    
    def calculate_operational_efficiency(self, queue_times: List[float], space_utilization: float,
                                       baseline_queue_time: float = 10.0, 
                                       baseline_utilization: float = 0.75) -> Dict:
        """Calculate operational efficiency metrics"""
        
        # Queue time analysis
        if queue_times:
            avg_queue_time = np.mean(queue_times)
            max_queue_time = np.max(queue_times)
            queue_time_std = np.std(queue_times)
        else:
            avg_queue_time = max_queue_time = queue_time_std = 0
        
        # Queue time reduction
        queue_time_reduction = baseline_queue_time - avg_queue_time
        queue_time_reduction_pct = (queue_time_reduction / baseline_queue_time) * 100 if baseline_queue_time > 0 else 0
        
        # Space utilization improvement
        utilization_improvement = space_utilization - baseline_utilization
        utilization_improvement_pct = (utilization_improvement / baseline_utilization) * 100 if baseline_utilization > 0 else 0
        
        # Target achievements
        target_queue_reduction = self.kpi_targets['queue_time_reduction']
        target_utilization_improvement = self.kpi_targets['space_utilization_improvement']
        
        queue_target_achieved = queue_time_reduction_pct >= target_queue_reduction
        utilization_target_achieved = utilization_improvement_pct >= target_utilization_improvement
        
        # Overall efficiency score (0-100)
        queue_score = min(100, max(0, queue_time_reduction_pct / target_queue_reduction * 50))
        utilization_score = min(100, max(0, utilization_improvement_pct / target_utilization_improvement * 50))
        overall_efficiency = queue_score + utilization_score
        
        return {
            'baseline_queue_time': baseline_queue_time,
            'avg_queue_time': round(avg_queue_time, 1),
            'max_queue_time': round(max_queue_time, 1),
            'queue_time_std': round(queue_time_std, 1),
            'queue_time_reduction_pct': round(queue_time_reduction_pct, 1),
            'baseline_utilization': round(baseline_utilization * 100, 1),
            'current_utilization': round(space_utilization * 100, 1),
            'utilization_improvement_pct': round(utilization_improvement_pct, 1),
            'queue_target_achieved': queue_target_achieved,
            'utilization_target_achieved': utilization_target_achieved,
            'overall_efficiency_score': round(overall_efficiency, 1)
        }
    
    def calculate_digital_engagement(self, total_passengers: int, app_users: int, 
                                   digital_touchpoints: int, baseline_adoption: float = 0.20) -> Dict:
        """Calculate digital engagement and app adoption metrics"""
        
        # App adoption calculation
        app_adoption_rate = app_users / total_passengers if total_passengers > 0 else 0
        adoption_uplift = app_adoption_rate - baseline_adoption
        adoption_uplift_pct = (adoption_uplift / baseline_adoption) * 100 if baseline_adoption > 0 else 0
        
        # Digital engagement
        touchpoints_per_user = digital_touchpoints / app_users if app_users > 0 else 0
        touchpoints_per_passenger = digital_touchpoints / total_passengers if total_passengers > 0 else 0
        
        # Target achievement
        target_adoption = self.kpi_targets['app_adoption_target'] / 100
        target_achieved = app_adoption_rate >= target_adoption
        
        # Engagement level assessment
        if app_adoption_rate >= target_adoption:
            engagement_level = 'Excellent'
        elif app_adoption_rate >= baseline_adoption * 1.5:
            engagement_level = 'Good'
        elif app_adoption_rate >= baseline_adoption:
            engagement_level = 'Average'
        else:
            engagement_level = 'Poor'
        
        return {
            'total_passengers': total_passengers,
            'app_users': app_users,
            'baseline_adoption_rate': round(baseline_adoption * 100, 1),
            'current_adoption_rate': round(app_adoption_rate * 100, 1),
            'target_adoption_rate': round(target_adoption * 100, 1),
            'adoption_uplift_pct': round(adoption_uplift_pct, 1),
            'digital_touchpoints': digital_touchpoints,
            'touchpoints_per_user': round(touchpoints_per_user, 1),
            'touchpoints_per_passenger': round(touchpoints_per_passenger, 2),
            'engagement_level': engagement_level,
            'target_achieved': target_achieved
        }
    
    def calculate_roi_metrics(self, airport_code: str, investment_amount: float,
                            annual_revenue_increase: float, operational_savings: float = 0,
                            implementation_period_months: int = 12) -> Dict:
        """Calculate return on investment metrics"""
        
        # Annual benefits
        total_annual_benefit = annual_revenue_increase + operational_savings
        
        # ROI calculations
        simple_roi = (total_annual_benefit / investment_amount) * 100 if investment_amount > 0 else 0
        
        # Payback period (in months)
        monthly_benefit = total_annual_benefit / 12
        payback_period = investment_amount / monthly_benefit if monthly_benefit > 0 else float('inf')
        
        # NPV calculation (assuming 10% discount rate over 5 years)
        discount_rate = 0.10
        npv = 0
        for year in range(1, 6):
            npv += total_annual_benefit / ((1 + discount_rate) ** year)
        npv -= investment_amount
        
        # IRR estimation (simplified)
        irr_estimate = (total_annual_benefit / investment_amount) * 100 if investment_amount > 0 else 0
        
        # ROI category
        if simple_roi >= 200:
            roi_category = 'Excellent'
        elif simple_roi >= 100:
            roi_category = 'Very Good'
        elif simple_roi >= 50:
            roi_category = 'Good'
        elif simple_roi >= 25:
            roi_category = 'Fair'
        else:
            roi_category = 'Poor'
        
        return {
            'investment_amount': investment_amount,
            'investment_amount_cr': round(investment_amount / 10000000, 1),
            'annual_revenue_increase': annual_revenue_increase,
            'operational_savings': operational_savings,
            'total_annual_benefit': total_annual_benefit,
            'total_annual_benefit_cr': round(total_annual_benefit / 10000000, 1),
            'simple_roi': round(simple_roi, 1),
            'payback_period_months': round(payback_period, 1) if payback_period != float('inf') else 'N/A',
            'npv': round(npv, 0),
            'npv_cr': round(npv / 10000000, 1),
            'irr_estimate': round(irr_estimate, 1),
            'roi_category': roi_category,
            'implementation_period_months': implementation_period_months
        }
    
    def calculate_comprehensive_kpis(self, airport_code: str, performance_data: Dict) -> Dict:
        """Calculate comprehensive KPI dashboard with all metrics"""
        
        results = {
            'airport_code': airport_code,
            'airport_name': self.airport_profiles[airport_code]['name'],
            'calculation_timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Revenue metrics
        if 'revenue' in performance_data:
            revenue_data = performance_data['revenue']
            results['metrics']['revenue'] = self.calculate_revenue_uplift(
                airport_code,
                revenue_data.get('current_revenue_per_pax', 0),
                revenue_data.get('baseline_revenue_per_pax')
            )
        
        # NPS metrics
        if 'nps' in performance_data:
            nps_data = performance_data['nps']
            results['metrics']['nps'] = self.calculate_nps_metrics(
                airport_code,
                nps_data.get('current_nps', 0),
                nps_data.get('promoters', 0),
                nps_data.get('passives', 0),
                nps_data.get('detractors', 0)
            )
        
        # Dwell time metrics
        if 'dwell_time' in performance_data:
            dwell_data = performance_data['dwell_time']
            results['metrics']['dwell_time'] = self.calculate_dwell_time_metrics(
                airport_code,
                dwell_data.get('avg_dwell_time', 0),
                dwell_data.get('variance', 0)
            )
        
        # Conversion metrics
        if 'conversion' in performance_data:
            conv_data = performance_data['conversion']
            results['metrics']['conversion'] = self.calculate_conversion_metrics(
                conv_data.get('total_passengers', 0),
                conv_data.get('purchasing_passengers', 0),
                conv_data.get('total_revenue', 0),
                conv_data.get('baseline_conversion_rate', 0.15)
            )
        
        # Operational efficiency
        if 'operations' in performance_data:
            ops_data = performance_data['operations']
            results['metrics']['operations'] = self.calculate_operational_efficiency(
                ops_data.get('queue_times', []),
                ops_data.get('space_utilization', 0.75),
                ops_data.get('baseline_queue_time', 10.0),
                ops_data.get('baseline_utilization', 0.75)
            )
        
        # Digital engagement
        if 'digital' in performance_data:
            digital_data = performance_data['digital']
            results['metrics']['digital'] = self.calculate_digital_engagement(
                digital_data.get('total_passengers', 0),
                digital_data.get('app_users', 0),
                digital_data.get('digital_touchpoints', 0),
                digital_data.get('baseline_adoption', 0.20)
            )
        
        # ROI metrics
        if 'roi' in performance_data:
            roi_data = performance_data['roi']
            results['metrics']['roi'] = self.calculate_roi_metrics(
                airport_code,
                roi_data.get('investment_amount', 0),
                roi_data.get('annual_revenue_increase', 0),
                roi_data.get('operational_savings', 0),
                roi_data.get('implementation_period_months', 12)
            )
        
        # Overall performance score (0-100)
        performance_scores = []
        
        if 'revenue' in results['metrics']:
            performance_scores.append(min(100, results['metrics']['revenue']['progress_to_target']))
        
        if 'nps' in results['metrics']:
            performance_scores.append(min(100, results['metrics']['nps']['progress_to_target']))
        
        if 'dwell_time' in results['metrics']:
            performance_scores.append(min(100, results['metrics']['dwell_time']['progress_to_target']))
        
        if 'operations' in results['metrics']:
            performance_scores.append(results['metrics']['operations']['overall_efficiency_score'])
        
        if performance_scores:
            results['overall_performance_score'] = round(np.mean(performance_scores), 1)
        else:
            results['overall_performance_score'] = 0
        
        return results
    
    def compare_airports(self, del_data: Dict, jai_data: Dict) -> Dict:
        """Compare performance between DEL and JAI airports"""
        
        comparison = {
            'comparison_timestamp': datetime.now().isoformat(),
            'airports': {
                'DEL': del_data,
                'JAI': jai_data
            },
            'analysis': {}
        }
        
        # Revenue comparison
        if 'revenue' in del_data.get('metrics', {}) and 'revenue' in jai_data.get('metrics', {}):
            del_revenue = del_data['metrics']['revenue']
            jai_revenue = jai_data['metrics']['revenue']
            
            comparison['analysis']['revenue'] = {
                'del_uplift_pct': del_revenue['percentage_uplift'],
                'jai_uplift_pct': jai_revenue['percentage_uplift'],
                'del_annual_impact_cr': del_revenue['annual_impact_cr'],
                'jai_annual_impact_cr': jai_revenue['annual_impact_cr'],
                'better_performer': 'DEL' if del_revenue['percentage_uplift'] > jai_revenue['percentage_uplift'] else 'JAI',
                'total_impact_cr': del_revenue['annual_impact_cr'] + jai_revenue['annual_impact_cr']
            }
        
        # NPS comparison
        if 'nps' in del_data.get('metrics', {}) and 'nps' in jai_data.get('metrics', {}):
            del_nps = del_data['metrics']['nps']
            jai_nps = jai_data['metrics']['nps']
            
            comparison['analysis']['nps'] = {
                'del_current_nps': del_nps['current_nps'],
                'jai_current_nps': jai_nps['current_nps'],
                'del_improvement': del_nps['nps_improvement'],
                'jai_improvement': jai_nps['nps_improvement'],
                'better_performer': 'DEL' if del_nps['current_nps'] > jai_nps['current_nps'] else 'JAI',
                'avg_nps_improvement': round((del_nps['nps_improvement'] + jai_nps['nps_improvement']) / 2, 1)
            }
        
        # Overall performance comparison
        del_score = del_data.get('overall_performance_score', 0)
        jai_score = jai_data.get('overall_performance_score', 0)
        
        comparison['analysis']['overall'] = {
            'del_performance_score': del_score,
            'jai_performance_score': jai_score,
            'performance_leader': 'DEL' if del_score > jai_score else 'JAI',
            'performance_gap': abs(del_score - jai_score),
            'combined_performance': round((del_score + jai_score) / 2, 1)
        }
        
        return comparison
    
    def generate_kpi_summary(self, kpi_data: Dict) -> str:
        """Generate human-readable KPI summary"""
        
        airport_name = kpi_data.get('airport_name', 'Airport')
        overall_score = kpi_data.get('overall_performance_score', 0)
        
        summary = f"**{airport_name} Performance Summary**\n\n"
        summary += f"Overall Performance Score: {overall_score}/100\n\n"
        
        metrics = kpi_data.get('metrics', {})
        
        if 'revenue' in metrics:
            rev = metrics['revenue']
            summary += f"• **Revenue**: ₹{rev['current_revenue_per_pax']}/passenger "
            summary += f"({rev['percentage_uplift']:+.1f}% vs baseline, "
            summary += f"₹{rev['annual_impact_cr']:.1f} Cr annual impact)\n"
        
        if 'nps' in metrics:
            nps = metrics['nps']
            summary += f"• **NPS**: {nps['current_nps']:.1f} "
            summary += f"({nps['nps_improvement']:+.1f} pts improvement, "
            summary += f"{nps['nps_category']} category)\n"
        
        if 'dwell_time' in metrics:
            dwell = metrics['dwell_time']
            summary += f"• **Dwell Time**: {dwell['current_dwell_time']:.1f} min avg "
            summary += f"(±{dwell['current_variance']:.1f} min variance, "
            summary += f"{dwell['efficiency_status']})\n"
        
        if 'conversion' in metrics:
            conv = metrics['conversion']
            summary += f"• **Conversion**: {conv['current_conversion_rate']:.1f}% "
            summary += f"({conv['conversion_uplift_pct']:+.1f}% improvement)\n"
        
        if 'digital' in metrics:
            digital = metrics['digital']
            summary += f"• **Digital Engagement**: {digital['current_adoption_rate']:.1f}% app adoption "
            summary += f"({digital['engagement_level']} level)\n"
        
        if 'roi' in metrics:
            roi = metrics['roi']
            summary += f"• **ROI**: {roi['simple_roi']:.1f}% "
            summary += f"({roi['payback_period_months']} month payback, "
            summary += f"{roi['roi_category']} category)\n"
        
        return summary
