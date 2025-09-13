import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class BiasDetector:
    """Advanced bias detection engine for AI fairness auditing"""
    
    def __init__(self):
        self.fairness_thresholds = {
            'statistical_parity_difference': 0.1,  # 10% difference threshold
            'demographic_parity_ratio': 0.8,       # 80% rule
            'equal_opportunity_difference': 0.1,    # Equal opportunity threshold
            'equalized_odds_difference': 0.1        # Equalized odds threshold
        }
    
    def analyze_bias(self, df: pd.DataFrame, protected_attributes: List[str], target_variable: str) -> Dict[str, Any]:
        """Comprehensive bias analysis across multiple protected attributes"""
        
        print(f"🔍 Starting bias analysis...")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🛡️ Protected attributes: {protected_attributes}")
        print(f"🎯 Target variable: {target_variable}")
        
        # Clean and prepare data
        df_clean = self._prepare_data(df, protected_attributes, target_variable)
        
        # Initialize results
        bias_results = {
            'bias_metrics': {},
            'fairness_violations': [],
            'overall_statistics': self._calculate_overall_stats(df_clean, target_variable)
        }
        
        # Analyze each protected attribute
        for attr in protected_attributes:
            if attr in df_clean.columns:
                print(f"🔍 Analyzing bias for: {attr}")
                
                attr_analysis = self._analyze_attribute_bias(
                    df_clean, attr, target_variable
                )
                
                bias_results['bias_metrics'][attr] = attr_analysis
                
                # Check for violations
                violations = self._detect_fairness_violations(attr_analysis, attr)
                bias_results['fairness_violations'].extend(violations)
            else:
                print(f"⚠️ Warning: Attribute '{attr}' not found in dataset")
        
        # Intersectional bias analysis (if multiple attributes)
        if len(protected_attributes) > 1:
            intersectional_analysis = self._analyze_intersectional_bias(
                df_clean, protected_attributes, target_variable
            )
            bias_results['intersectional_bias'] = intersectional_analysis
        
        print(f"✅ Bias analysis complete. Found {len(bias_results['fairness_violations'])} violations.")
        
        return bias_results
    
    def _prepare_data(self, df: pd.DataFrame, protected_attributes: List[str], target_variable: str) -> pd.DataFrame:
        """Clean and prepare data for bias analysis"""
        
        df_clean = df.copy()
        
        # Handle missing values in target variable
        if df_clean[target_variable].isnull().sum() > 0:
            print(f"⚠️ Removing {df_clean[target_variable].isnull().sum()} rows with missing target values")
            df_clean = df_clean.dropna(subset=[target_variable])
        
        # Ensure target variable is binary (0/1)
        unique_targets = df_clean[target_variable].unique()
        if len(unique_targets) == 2:
            # Convert to 0/1 if needed
            if not set(unique_targets).issubset({0, 1}):
                target_mapping = {unique_targets[0]: 0, unique_targets[1]: 1}
                df_clean[target_variable] = df_clean[target_variable].map(target_mapping)
                print(f"📝 Mapped target variable: {target_mapping}")
        else:
            print(f"⚠️ Warning: Target variable has {len(unique_targets)} unique values. Expected binary.")
        
        # Handle missing values in protected attributes
        for attr in protected_attributes:
            if attr in df_clean.columns:
                missing_count = df_clean[attr].isnull().sum()
                if missing_count > 0:
                    print(f"⚠️ {missing_count} missing values in {attr}, filling with 'Unknown'")
                    df_clean[attr] = df_clean[attr].fillna('Unknown')
        
        return df_clean
    
    def _calculate_overall_stats(self, df: pd.DataFrame, target_variable: str) -> Dict[str, Any]:
        """Calculate overall dataset statistics"""
        
        total_records = len(df)
        positive_outcomes = df[target_variable].sum()
        positive_rate = positive_outcomes / total_records if total_records > 0 else 0
        
        return {
            'total_records': total_records,
            'positive_outcomes': int(positive_outcomes),
            'overall_positive_rate': positive_rate,
            'negative_outcomes': total_records - int(positive_outcomes),
            'overall_negative_rate': 1 - positive_rate
        }
    
    def _analyze_attribute_bias(self, df: pd.DataFrame, attribute: str, target_variable: str) -> Dict[str, Any]:
        """Analyze bias for a single protected attribute"""
        
        # Group statistics
        groups = {}
        group_stats = df.groupby(attribute).agg({
            target_variable: ['count', 'sum', 'mean']
        }).round(4)
        
        group_stats.columns = ['group_size', 'positive_outcomes', 'positive_rate']
        
        total_records = len(df)
        
        for group_name, stats in group_stats.iterrows():
            groups[str(group_name)] = {
                'group_size': int(stats['group_size']),
                'positive_outcomes': int(stats['positive_outcomes']),
                'positive_rate': float(stats['positive_rate']),
                'negative_outcomes': int(stats['group_size'] - stats['positive_outcomes']),
                'representation': float(stats['group_size']) / total_records,
                'sample_adequacy': 'adequate' if stats['group_size'] >= 30 else 'small_sample'
            }
        
        # Calculate bias metrics
        disparities = self._calculate_disparities(groups)
        fairness_metrics = self._calculate_fairness_metrics(groups, disparities)
        
        return {
            'groups': groups,
            'disparities': disparities,
            'fairness_metrics': fairness_metrics,
            'statistical_summary': self._generate_statistical_summary(groups)
        }
    
    def _calculate_disparities(self, groups: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate various disparity metrics"""
        
        if len(groups) < 2:
            return {
                'statistical_parity_difference': 0.0,
                'demographic_parity_ratio': 1.0,
                'equal_opportunity_difference': 0.0,
                'max_disparity': 0.0,
                'min_disparity': 0.0
            }
        
        # Get positive rates for all groups
        positive_rates = [group['positive_rate'] for group in groups.values()]
        
        # Statistical Parity Difference (max - min positive rate)
        max_rate = max(positive_rates)
        min_rate = min(positive_rates)
        statistical_parity_difference = max_rate - min_rate
        
        # Demographic Parity Ratio (min / max positive rate)
        demographic_parity_ratio = min_rate / max_rate if max_rate > 0 else 1.0
        
        # Equal Opportunity Difference (for simplicity, same as statistical parity here)
        equal_opportunity_difference = statistical_parity_difference
        
        return {
            'statistical_parity_difference': statistical_parity_difference,
            'demographic_parity_ratio': demographic_parity_ratio,
            'equal_opportunity_difference': equal_opportunity_difference,
            'max_disparity': statistical_parity_difference,
            'min_disparity': 0.0,
            'rate_range': {
                'max_positive_rate': max_rate,
                'min_positive_rate': min_rate,
                'rate_spread': max_rate - min_rate
            }
        }
    
    def _calculate_fairness_metrics(self, groups: Dict[str, Dict], disparities: Dict[str, float]) -> Dict[str, Any]:
        """Calculate fairness test results"""
        
        # 80% Rule (Demographic Parity)
        passes_80_percent_rule = disparities['demographic_parity_ratio'] >= 0.8
        
        # Statistical Parity Test
        passes_statistical_parity = disparities['statistical_parity_difference'] <= self.fairness_thresholds['statistical_parity_difference']
        
        # Equal Opportunity Test
        passes_equal_opportunity = disparities['equal_opportunity_difference'] <= self.fairness_thresholds['equal_opportunity_difference']
        
        return {
            'passes_80_percent_rule': passes_80_percent_rule,
            'passes_statistical_parity': passes_statistical_parity,
            'passes_equal_opportunity': passes_equal_opportunity,
            'overall_fair': passes_80_percent_rule and passes_statistical_parity,
            'fairness_score': self._calculate_fairness_score(disparities),
            'bias_severity': self._assess_bias_severity(disparities)
        }
    
    def _calculate_fairness_score(self, disparities: Dict[str, float]) -> float:
        """Calculate an overall fairness score (0-1, higher is more fair)"""
        
        spd = disparities['statistical_parity_difference']
        dpr = disparities['demographic_parity_ratio']
        
        # Combine metrics into a single score
        spd_score = max(0, 1 - (spd / 0.5))  # Normalize to 0-1
        dpr_score = min(dpr / 0.8, 1.0)      # 80% rule normalized
        
        # Weighted average
        fairness_score = (spd_score * 0.6) + (dpr_score * 0.4)
        
        return round(fairness_score, 3)
    
    def _assess_bias_severity(self, disparities: Dict[str, float]) -> str:
        """Assess the severity of bias"""
        
        spd = disparities['statistical_parity_difference']
        dpr = disparities['demographic_parity_ratio']
        
        if spd >= 0.3 or dpr <= 0.5:
            return 'severe'
        elif spd >= 0.2 or dpr <= 0.6:
            return 'high'
        elif spd >= 0.1 or dpr <= 0.8:
            return 'moderate'
        elif spd >= 0.05 or dpr <= 0.9:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_statistical_summary(self, groups: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate statistical summary of group differences"""
        
        group_names = list(groups.keys())
        group_sizes = [group['group_size'] for group in groups.values()]
        positive_rates = [group['positive_rate'] for group in groups.values()]
        
        return {
            'number_of_groups': len(groups),
            'largest_group': group_names[np.argmax(group_sizes)],
            'smallest_group': group_names[np.argmin(group_sizes)],
            'highest_approval_group': group_names[np.argmax(positive_rates)],
            'lowest_approval_group': group_names[np.argmin(positive_rates)],
            'size_imbalance_ratio': max(group_sizes) / min(group_sizes) if min(group_sizes) > 0 else float('inf'),
            'approval_rate_variance': np.var(positive_rates)
        }
    
    def _detect_fairness_violations(self, analysis: Dict[str, Any], attribute: str) -> List[Dict[str, Any]]:
        """Detect specific fairness violations"""
        
        violations = []
        disparities = analysis['disparities']
        fairness_metrics = analysis['fairness_metrics']
        groups = analysis['groups']
        
        # Statistical Parity Violation
        if disparities['statistical_parity_difference'] > self.fairness_thresholds['statistical_parity_difference']:
            severity = 'high' if disparities['statistical_parity_difference'] > 0.2 else 'medium'
            violations.append({
                'type': 'statistical_parity_violation',
                'attribute': attribute,
                'severity': severity,
                'metric_value': disparities['statistical_parity_difference'],
                'threshold': self.fairness_thresholds['statistical_parity_difference'],
                'description': f"Statistical parity difference of {disparities['statistical_parity_difference']:.1%} exceeds threshold of {self.fairness_thresholds['statistical_parity_difference']:.1%}"
            })
        
        # 80% Rule Violation
        if not fairness_metrics['passes_80_percent_rule']:
            violations.append({
                'type': 'demographic_parity_violation',
                'attribute': attribute,
                'severity': 'high' if disparities['demographic_parity_ratio'] < 0.6 else 'medium',
                'metric_value': disparities['demographic_parity_ratio'],
                'threshold': 0.8,
                'description': f"Demographic parity ratio of {disparities['demographic_parity_ratio']:.3f} fails the 80% rule"
            })
        
        # Sample Size Warnings
        small_samples = [name for name, group in groups.items() if group['sample_adequacy'] == 'small_sample']
        if small_samples:
            violations.append({
                'type': 'small_sample_warning',
                'attribute': attribute,
                'severity': 'low',
                'metric_value': len(small_samples),
                'threshold': 30,
                'description': f"Groups with small sample sizes (< 30): {', '.join(small_samples)}"
            })
        
        # Extreme Bias Warning
        if disparities['statistical_parity_difference'] > 0.5:
            violations.append({
                'type': 'extreme_bias_warning',
                'attribute': attribute,
                'severity': 'critical',
                'metric_value': disparities['statistical_parity_difference'],
                'threshold': 0.5,
                'description': f"Extreme bias detected: {disparities['statistical_parity_difference']:.1%} difference suggests systematic discrimination"
            })
        
        return violations
    
    def _analyze_intersectional_bias(self, df: pd.DataFrame, protected_attributes: List[str], target_variable: str) -> Dict[str, Any]:
        """Analyze intersectional bias across multiple protected attributes"""
        
        if len(protected_attributes) < 2:
            return {}
        
        print("🔍 Analyzing intersectional bias...")
        
        # Create intersection groups
        df['intersection_group'] = df[protected_attributes].apply(
            lambda x: ' & '.join([f"{attr}:{val}" for attr, val in zip(protected_attributes, x)]), 
            axis=1
        )
        
        # Analyze intersection groups
        intersection_analysis = self._analyze_attribute_bias(df, 'intersection_group', target_variable)
        
        # Find most/least advantaged intersections
        groups = intersection_analysis['groups']
        if groups:
            positive_rates = {group: stats['positive_rate'] for group, stats in groups.items()}
            
            most_advantaged = max(positive_rates, key=positive_rates.get)
            least_advantaged = min(positive_rates, key=positive_rates.get)
            
            intersectional_summary = {
                'total_intersections': len(groups),
                'most_advantaged_group': most_advantaged,
                'least_advantaged_group': least_advantaged,
                'max_intersectional_disparity': positive_rates[most_advantaged] - positive_rates[least_advantaged],
                'intersectional_groups': groups
            }
            
            return intersectional_summary
        
        return {}
    
    def generate_bias_summary(self, bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level bias summary for reporting"""
        
        total_violations = len(bias_results['fairness_violations'])
        
        # Categorize violations by severity
        violation_severity = defaultdict(int)
        for violation in bias_results['fairness_violations']:
            violation_severity[violation['severity']] += 1
        
        # Calculate overall bias score
        overall_bias_score = self._calculate_overall_bias_score(bias_results)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_bias_score, total_violations)
        
        return {
            'overall_bias_score': overall_bias_score,
            'risk_level': risk_level,
            'total_violations': total_violations,
            'violations_by_severity': dict(violation_severity),
            'attributes_analyzed': len(bias_results['bias_metrics']),
            'requires_immediate_attention': risk_level in ['CRITICAL', 'HIGH']
        }
    
    def _calculate_overall_bias_score(self, bias_results: Dict[str, Any]) -> float:
        """Calculate overall bias score across all attributes"""
        
        if not bias_results['bias_metrics']:
            return 0.0
        
        # Average fairness scores across attributes
        fairness_scores = []
        for attr_analysis in bias_results['bias_metrics'].values():
            fairness_score = attr_analysis['fairness_metrics']['fairness_score']
            fairness_scores.append(fairness_score)
        
        # Convert fairness score to bias score (inverse)
        avg_fairness = np.mean(fairness_scores)
        bias_score = 1 - avg_fairness
        
        # Adjust based on violation severity
        violation_penalty = min(len(bias_results['fairness_violations']) * 0.1, 0.5)
        
        final_bias_score = min(bias_score + violation_penalty, 1.0)
        
        return round(final_bias_score, 3)
    
    def _determine_risk_level(self, bias_score: float, violation_count: int) -> str:
        """Determine overall risk level"""
        
        if bias_score >= 0.7 or violation_count >= 3:
            return 'CRITICAL'
        elif bias_score >= 0.5 or violation_count >= 2:
            return 'HIGH'
        elif bias_score >= 0.3 or violation_count >= 1:
            return 'MEDIUM'
        elif bias_score >= 0.1:
            return 'LOW'
        else:
            return 'MINIMAL'
