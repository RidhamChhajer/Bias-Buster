import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)

class CustomBiasDetector:
    """Custom bias detection engine for Python 3.13 compatibility"""
    
    def __init__(self):
        self.supported_metrics = [
            'demographic_parity',
            'equalized_odds', 
            'statistical_parity',
            'representation_balance'
        ]
    
    def detect_bias(self, data: pd.DataFrame, target_col: str, 
                   protected_attrs: List[str]) -> Dict[str, Any]:
        """
        Comprehensive bias detection across multiple protected attributes
        
        Args:
            data: DataFrame containing the dataset
            target_col: Name of the target/outcome column
            protected_attrs: List of protected attribute column names
            
        Returns:
            Dictionary containing bias analysis results
        """
        try:
            results = {
                'summary': {
                    'total_records': len(data),
                    'target_variable': target_col,
                    'protected_attributes': protected_attrs,
                    'analysis_timestamp': pd.Timestamp.now().isoformat()
                },
                'bias_metrics': {},
                'group_statistics': {},
                'fairness_violations': []
            }
            
            # Analyze each protected attribute
            for attr in protected_attrs:
                if attr not in data.columns:
                    logger.warning(f"Protected attribute '{attr}' not found in data")
                    continue
                
                attr_results = self._analyze_protected_attribute(
                    data, target_col, attr
                )
                results['bias_metrics'][attr] = attr_results
                
                # Check for fairness violations
                violations = self._detect_fairness_violations(attr_results, attr)
                if violations:
                    results['fairness_violations'].extend(violations)
            
            # Calculate overall bias score
            results['overall_bias_score'] = self._calculate_overall_bias_score(
                results['bias_metrics']
            )
            
            # Determine risk level
            results['risk_assessment'] = self._assess_risk_level(
                results['overall_bias_score'], 
                results['fairness_violations']
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bias detection: {str(e)}")
            return {
                'error': str(e),
                'summary': {'total_records': 0, 'analysis_failed': True}
            }
    
    def _analyze_protected_attribute(self, data: pd.DataFrame, 
                                   target_col: str, attr: str) -> Dict[str, Any]:
        """Analyze bias for a specific protected attribute"""
        
        # Get unique groups in the protected attribute
        groups = data[attr].unique()
        group_stats = {}
        
        for group in groups:
            group_data = data[data[attr] == group]
            
            # Basic statistics
            group_size = len(group_data)
            positive_outcomes = group_data[target_col].sum() if group_data[target_col].dtype in ['int64', 'float64'] else len(group_data[group_data[target_col] == group_data[target_col].mode()[0]])
            
            # Calculate rates
            positive_rate = positive_outcomes / group_size if group_size > 0 else 0
            representation = group_size / len(data)
            
            group_stats[str(group)] = {
                'group_size': int(group_size),
                'positive_outcomes': int(positive_outcomes),
                'positive_rate': round(float(positive_rate), 4),
                'representation': round(float(representation), 4),
                'sample_adequacy': 'adequate' if group_size >= 30 else 'small_sample'
            }
        
        # Calculate disparities
        disparities = self._calculate_disparities(group_stats)
        
        return {
            'attribute': attr,
            'groups': group_stats,
            'disparities': disparities,
            'fairness_metrics': self._calculate_fairness_metrics(group_stats)
        }
    
    def _calculate_disparities(self, group_stats: Dict) -> Dict[str, float]:
        """Calculate disparity measures between groups"""
        
        positive_rates = [stats['positive_rate'] for stats in group_stats.values()]
        representations = [stats['representation'] for stats in group_stats.values()]
        
        if len(positive_rates) < 2:
            return {'insufficient_groups': True}
        
        # Statistical parity difference
        max_positive_rate = max(positive_rates)
        min_positive_rate = min(positive_rates)
        parity_difference = max_positive_rate - min_positive_rate
        
        # Representation balance
        max_representation = max(representations)
        min_representation = min(representations)
        representation_imbalance = max_representation - min_representation
        
        return {
            'statistical_parity_difference': round(parity_difference, 4),
            'max_positive_rate': round(max_positive_rate, 4),
            'min_positive_rate': round(min_positive_rate, 4),
            'representation_imbalance': round(representation_imbalance, 4),
            'demographic_parity_ratio': round(min_positive_rate / max_positive_rate if max_positive_rate > 0 else 0, 4)
        }
    
    def _calculate_fairness_metrics(self, group_stats: Dict) -> Dict[str, Any]:
        """Calculate standard fairness metrics"""
        
        positive_rates = [stats['positive_rate'] for stats in group_stats.values()]
        
        if len(positive_rates) < 2:
            return {'error': 'Insufficient groups for fairness calculation'}
        
        # 80% rule check (demographic parity)
        min_rate = min(positive_rates)
        max_rate = max(positive_rates)
        passes_80_rule = (min_rate / max_rate) >= 0.8 if max_rate > 0 else False
        
        # Coefficient of variation
        mean_rate = np.mean(positive_rates)
        std_rate = np.std(positive_rates)
        cv = std_rate / mean_rate if mean_rate > 0 else float('inf')
        
        return {
            'passes_80_percent_rule': passes_80_rule,
            'demographic_parity_ratio': round(min_rate / max_rate if max_rate > 0 else 0, 4),
            'coefficient_of_variation': round(cv, 4),
            'rate_stability': 'stable' if cv < 0.1 else 'moderate' if cv < 0.3 else 'high_variation'
        }
    
    def _detect_fairness_violations(self, attr_results: Dict, attr_name: str) -> List[Dict]:
        """Detect potential fairness violations"""
        
        violations = []
        
        try:
            disparities = attr_results.get('disparities', {})
            fairness_metrics = attr_results.get('fairness_metrics', {})
            
            # Check statistical parity
            parity_diff = disparities.get('statistical_parity_difference', 0)
            if parity_diff > 0.1:  # 10% threshold
                violations.append({
                    'type': 'statistical_parity_violation',
                    'attribute': attr_name,
                    'severity': 'high' if parity_diff > 0.2 else 'medium',
                    'description': f'Large disparity in positive rates ({parity_diff:.1%}) across {attr_name} groups',
                    'metric_value': parity_diff
                })
            
            # Check 80% rule
            if not fairness_metrics.get('passes_80_percent_rule', True):
                violations.append({
                    'type': 'demographic_parity_violation',
                    'attribute': attr_name,
                    'severity': 'high',
                    'description': f'Fails 80% rule for demographic parity in {attr_name}',
                    'metric_value': fairness_metrics.get('demographic_parity_ratio', 0)
                })
            
            # Check representation imbalance
            repr_imbalance = disparities.get('representation_imbalance', 0)
            if repr_imbalance > 0.3:  # 30% threshold
                violations.append({
                    'type': 'representation_imbalance',
                    'attribute': attr_name,
                    'severity': 'medium',
                    'description': f'Significant representation imbalance ({repr_imbalance:.1%}) in {attr_name} groups',
                    'metric_value': repr_imbalance
                })
                
        except Exception as e:
            logger.error(f"Error detecting violations for {attr_name}: {str(e)}")
        
        return violations
    
    def _calculate_overall_bias_score(self, bias_metrics: Dict) -> float:
        """Calculate an overall bias score from 0 (no bias) to 1 (high bias)"""
        
        if not bias_metrics:
            return 0.0
        
        scores = []
        
        for attr, metrics in bias_metrics.items():
            disparities = metrics.get('disparities', {})
            
            # Weight different disparity measures
            parity_score = disparities.get('statistical_parity_difference', 0) * 2
            ratio_score = (1 - disparities.get('demographic_parity_ratio', 1)) * 1.5
            repr_score = disparities.get('representation_imbalance', 0) * 1
            
            attr_score = min(parity_score + ratio_score + repr_score, 1.0)
            scores.append(attr_score)
        
        return round(np.mean(scores), 3) if scores else 0.0
    
    def _assess_risk_level(self, bias_score: float, violations: List) -> Dict[str, Any]:
        """Assess overall risk level based on bias score and violations"""
        
        high_severity_violations = len([v for v in violations if v.get('severity') == 'high'])
        medium_severity_violations = len([v for v in violations if v.get('severity') == 'medium'])
        
        # Determine risk level
        if bias_score >= 0.7 or high_severity_violations >= 2:
            risk_level = 'CRITICAL'
            risk_color = '#dc3545'  # Red
        elif bias_score >= 0.5 or high_severity_violations >= 1:
            risk_level = 'HIGH'
            risk_color = '#fd7e14'  # Orange
        elif bias_score >= 0.3 or medium_severity_violations >= 2:
            risk_level = 'MEDIUM'
            risk_color = '#ffc107'  # Yellow
        elif bias_score >= 0.1 or medium_severity_violations >= 1:
            risk_level = 'LOW'
            risk_color = '#17a2b8'  # Blue
        else:
            risk_level = 'MINIMAL'
            risk_color = '#28a745'  # Green
        
        return {
            'level': risk_level,
            'score': bias_score,
            'color': risk_color,
            'total_violations': len(violations),
            'high_severity_violations': high_severity_violations,
            'medium_severity_violations': medium_severity_violations,
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        
        recommendations = {
            'CRITICAL': 'Immediate action required. Do not deploy this model without bias mitigation.',
            'HIGH': 'Significant bias detected. Implement bias mitigation strategies before deployment.',
            'MEDIUM': 'Moderate bias present. Consider bias reduction techniques and monitoring.',
            'LOW': 'Low bias detected. Monitor for bias in production and consider minor adjustments.',
            'MINIMAL': 'Minimal bias detected. Continue monitoring and maintain current practices.'
        }
        
        return recommendations.get(risk_level, 'Unknown risk level')
