import openai
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class OpenAIService:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        openai.api_key = self.api_key
        self.model = os.getenv('OPENAI_MODEL', 'gpt-5-nano-2025-08-07')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
        
        # Enhanced Bias Auditor System Prompt
        self.bias_auditor_prompt = """You are Bias Buster, an expert AI auditor and fairness consultant with 15+ years of experience in AI ethics and bias detection.

Your role is to detect, explain, and suggest fixes for bias in datasets and AI models. You must evaluate datasets and outputs not just mathematically, but also contextually and ethically.

EXPERTISE AREAS:
- Statistical bias detection (demographic parity, equalized odds, calibration)
- Legal compliance (GDPR, ECOA, fair lending laws, employment discrimination)
- Intersectional bias analysis (multiple protected attributes)
- Real-world impact assessment
- Remediation strategies and implementation

When given a dataset analysis, you must:
1. Identify potential biases across gender, race, age, and other sensitive attributes
2. Compare subgroup performance (accuracy, false positives, acceptance rates)
3. Explain in plain language why the bias might exist and its real-world implications
4. Suggest specific, actionable remedies (data rebalancing, algorithmic fixes, policy changes)
5. Assess legal and ethical risks
6. Generate a comprehensive fairness assessment

OUTPUT FORMAT:
**🎯 BIAS VERDICT**: Clear determination (FAIR/UNFAIR) with risk level (MINIMAL/LOW/MEDIUM/HIGH/CRITICAL)

**📊 EVIDENCE**: Specific numbers, statistical disparities, and examples from the data

**⚖️ LEGAL & ETHICAL IMPACT**: Potential legal risks, regulatory violations, and ethical concerns

**🔧 SPECIFIC REMEDIATION STEPS**: Concrete actions prioritized by urgency and impact

**📋 MONITORING RECOMMENDATIONS**: Ongoing measures to prevent future bias

**💡 BUSINESS CONTEXT**: Explanation of why this matters for the organization

RESPONSE STYLE:
- Professional yet accessible to non-technical stakeholders
- Data-driven with specific metrics and examples
- Actionable recommendations with timelines and priorities
- Consider intersectional effects and real-world consequences
- Always provide comprehensive analysis, even with limited data
- Never dismiss concerns - always provide reasoning and insights

CRITICAL: You must be thorough and honest about bias findings. Err on the side of caution when identifying potential discrimination."""

    def analyze_bias_with_context(self, bias_data: Dict[str, Any], dataset_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced bias analysis using expert auditor persona"""
        
        try:
            # Create comprehensive analysis prompt
            analysis_prompt = self._create_comprehensive_bias_prompt(bias_data, dataset_context)
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.bias_auditor_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_completion_tokens=self.max_tokens,
                reasoning_effort="medium",
                verbosity="high",
                temperature=0.3  # Lower temperature for more consistent analysis
            )
            
            ai_analysis = response.choices[0].message.content
            
            # Extract structured components from AI response
            analysis_components = self._parse_ai_analysis(ai_analysis)
            
            return {
                'ai_analysis': ai_analysis,
                'risk_level': analysis_components.get('risk_level', 'MEDIUM'),
                'bias_score': analysis_components.get('bias_score', 0.5),
                'recommendations': analysis_components.get('recommendations', []),
                'legal_risks': analysis_components.get('legal_risks', []),
                'model_used': self.model,
                'timestamp': datetime.now().isoformat(),
                'token_usage': {
                    'prompt_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                    'completion_tokens': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
                }
            }
            
        except Exception as e:
            print(f"Enhanced AI analysis failed: {str(e)}")
            # Return comprehensive fallback analysis
            return self._generate_expert_fallback_analysis(bias_data, dataset_context)
    
    def _create_comprehensive_bias_prompt(self, bias_data: Dict[str, Any], dataset_context: Dict[str, Any]) -> str:
        """Create detailed bias analysis prompt with full context"""
        
        dataset_info = dataset_context.get('dataset_info', {})
        violations = bias_data.get('fairness_violations', [])
        
        prompt = f"""
DATASET CONTEXT:
- Filename: {dataset_info.get('filename', 'Unknown')}
- Records: {dataset_info.get('total_records', 0):,}
- Target Variable: {dataset_info.get('target_variable', 'Unknown')} (what decision is being made)
- Protected Attributes: {', '.join(dataset_info.get('protected_attributes', []))}
- Use Case: {self._infer_use_case(dataset_info)}

BIAS ANALYSIS RESULTS:
"""
        
        # Add detailed metrics for each protected attribute
        for attr, metrics in bias_data.get('bias_metrics', {}).items():
            prompt += f"""
{attr.upper()} BIAS ANALYSIS:
Group Statistics:"""
            
            groups = metrics.get('groups', {})
            for group, stats in groups.items():
                prompt += f"""
  - {group}: {stats.get('group_size', 0):,} samples ({stats.get('representation', 0):.1%} of data)
    Positive rate: {stats.get('positive_rate', 0):.1%}"""
            
            disparities = metrics.get('disparities', {})
            fairness_metrics = metrics.get('fairness_metrics', {})
            
            prompt += f"""
  Statistical Parity Difference: {disparities.get('statistical_parity_difference', 0):.3f}
  Demographic Parity Ratio: {disparities.get('demographic_parity_ratio', 0):.3f}
  Passes 80% Rule: {fairness_metrics.get('passes_80_percent_rule', False)}
  Equal Opportunity Difference: {disparities.get('equal_opportunity_difference', 0):.3f}
"""
        
        # Add violations
        if violations:
            prompt += f"\nFAIRNESS VIOLATIONS DETECTED ({len(violations)}):\n"
            for violation in violations:
                prompt += f"- {violation.get('type', 'Unknown').title()}: {violation.get('description', '')}\n"
        
        prompt += """

ANALYSIS REQUEST:
As an expert AI bias auditor, provide a comprehensive analysis following your structured format. Consider:

1. The real-world implications of these findings
2. Legal and regulatory risks for this use case  
3. Intersectional effects if multiple protected attributes show bias
4. Specific remediation strategies with implementation priorities
5. Business impact and stakeholder concerns
6. Ongoing monitoring recommendations

Be thorough, specific, and actionable in your analysis. This analysis will be used for executive decision-making and compliance purposes.
"""
        
        return prompt
    
    def _infer_use_case(self, dataset_info: Dict) -> str:
        """Infer the likely use case from dataset characteristics"""
        
        target = dataset_info.get('target_variable', '').lower()
        filename = dataset_info.get('filename', '').lower()
        
        if any(word in target for word in ['hire', 'select', 'employ', 'recruit']):
            return "Employment/Hiring Decision"
        elif any(word in target for word in ['loan', 'credit', 'approve', 'lend']):
            return "Financial Services/Lending"
        elif any(word in target for word in ['admit', 'accept', 'enroll']):
            return "Education/Admissions"
        elif any(word in target for word in ['diagnose', 'treat', 'medical']):
            return "Healthcare/Medical Decision"
        elif any(word in filename for word in ['criminal', 'recid', 'justice']):
            return "Criminal Justice/Risk Assessment"
        else:
            return "Classification/Decision System"
    
    def _parse_ai_analysis(self, ai_analysis: str) -> Dict[str, Any]:
        """Extract structured components from AI analysis"""
        
        components = {
            'risk_level': 'MEDIUM',
            'bias_score': 0.5,
            'recommendations': [],
            'legal_risks': []
        }
        
        try:
            # Extract risk level
            if 'CRITICAL' in ai_analysis.upper():
                components['risk_level'] = 'CRITICAL'
                components['bias_score'] = 0.9
            elif 'HIGH' in ai_analysis.upper():
                components['risk_level'] = 'HIGH' 
                components['bias_score'] = 0.7
            elif 'MEDIUM' in ai_analysis.upper():
                components['risk_level'] = 'MEDIUM'
                components['bias_score'] = 0.5
            elif 'LOW' in ai_analysis.upper():
                components['risk_level'] = 'LOW'
                components['bias_score'] = 0.3
            elif 'MINIMAL' in ai_analysis.upper():
                components['risk_level'] = 'MINIMAL'
                components['bias_score'] = 0.1
            
            # Extract recommendations (look for numbered lists or bullet points)
            import re
            rec_patterns = [
                r'^\d+\.\s*(.+)$',  # Numbered lists
                r'^[-•]\s*(.+)$',   # Bullet points
                r'RECOMMEND[ED]*:\s*(.+)$',  # Recommendation headers
            ]
            
            recommendations = []
            for line in ai_analysis.split('\n'):
                for pattern in rec_patterns:
                    match = re.match(pattern, line.strip(), re.IGNORECASE)
                    if match and len(match.group(1)) > 10:  # Meaningful length
                        recommendations.append(match.group(1).strip())
            
            components['recommendations'] = recommendations[:5]  # Top 5 recommendations
            
            # Extract legal risks
            legal_keywords = ['legal', 'lawsuit', 'compliance', 'regulation', 'discriminat', 'GDPR', 'ECOA']
            legal_risks = []
            
            for line in ai_analysis.split('\n'):
                if any(keyword.lower() in line.lower() for keyword in legal_keywords):
                    if len(line.strip()) > 20:  # Meaningful content
                        legal_risks.append(line.strip())
            
            components['legal_risks'] = legal_risks[:3]  # Top 3 legal risks
            
        except Exception as e:
            print(f"Error parsing AI analysis: {e}")
        
        return components
    
    def _generate_expert_fallback_analysis(self, bias_data: Dict, dataset_context: Dict) -> Dict[str, Any]:
        """Generate expert-level fallback analysis when API fails"""
        
        # Analyze the data to provide expert insights
        violations = bias_data.get('fairness_violations', [])
        bias_metrics = bias_data.get('bias_metrics', {})
        
        # Calculate overall bias severity
        max_disparity = 0
        failed_80_rule = False
        
        for attr, metrics in bias_metrics.items():
            disparities = metrics.get('disparities', {})
            fairness = metrics.get('fairness_metrics', {})
            
            spd = abs(disparities.get('statistical_parity_difference', 0))
            if spd > max_disparity:
                max_disparity = spd
            
            if not fairness.get('passes_80_percent_rule', True):
                failed_80_rule = True
        
        # Determine risk level based on expert criteria
        if len(violations) >= 3 or max_disparity > 0.3:
            risk_level = 'CRITICAL'
            bias_score = 0.85
        elif len(violations) >= 2 or max_disparity > 0.2 or failed_80_rule:
            risk_level = 'HIGH'
            bias_score = 0.7
        elif len(violations) >= 1 or max_disparity > 0.1:
            risk_level = 'MEDIUM'
            bias_score = 0.5
        elif max_disparity > 0.05:
            risk_level = 'LOW'
            bias_score = 0.3
        else:
            risk_level = 'MINIMAL'
            bias_score = 0.1
        
        # Generate expert analysis
        dataset_info = dataset_context.get('dataset_info', {})
        use_case = self._infer_use_case(dataset_info)
        
        expert_analysis = f"""**🎯 BIAS VERDICT**: {'UNFAIR' if risk_level in ['CRITICAL', 'HIGH', 'MEDIUM'] else 'MOSTLY FAIR'} - {risk_level} Risk Level

As an expert AI auditor, I've identified {"significant" if risk_level in ['CRITICAL', 'HIGH'] else "moderate" if risk_level == 'MEDIUM' else "minimal"} bias concerns in your {use_case} system.

**📊 EVIDENCE**: 
- {len(violations)} fairness violations detected across protected attributes
- Maximum statistical parity difference: {max_disparity:.1%}
- {'Failed' if failed_80_rule else 'Passed'} the 80% rule test for demographic parity
- Analysis covers {len(bias_metrics)} protected attribute(s)

**⚖️ LEGAL & ETHICAL IMPACT**: 
{"This level of bias could violate anti-discrimination laws and create significant legal liability." if risk_level in ['CRITICAL', 'HIGH'] else "Some bias patterns present that should be addressed for compliance." if risk_level == 'MEDIUM' else "System appears mostly compliant with fairness standards."}

**🔧 SPECIFIC REMEDIATION STEPS**:
1. {"Immediate: Suspend system deployment until bias is mitigated" if risk_level == 'CRITICAL' else "Priority: Implement bias mitigation strategies" if risk_level in ['HIGH', 'MEDIUM'] else "Ongoing: Monitor for bias patterns"}
2. Data Level: Rebalance training data to ensure fair representation
3. Algorithm Level: Implement fairness constraints during model training
4. Process Level: Establish bias testing as standard practice
5. Monitoring: Set up ongoing bias detection systems

**📋 MONITORING RECOMMENDATIONS**: 
- {'Weekly' if risk_level == 'CRITICAL' else 'Monthly' if risk_level in ['HIGH', 'MEDIUM'] else 'Quarterly'} bias audits with these metrics
- Alert systems when disparities exceed {'5%' if risk_level in ['CRITICAL', 'HIGH'] else '10%'} threshold
- Regular fairness testing across all demographic groups
- Document all bias testing for compliance purposes

**💡 BUSINESS CONTEXT**: 
{risk_level.title()} bias in {use_case} systems can result in discriminatory outcomes, legal violations, and loss of stakeholder trust. {'Immediate action is required' if risk_level in ['CRITICAL', 'HIGH'] else 'Proactive improvements recommended' if risk_level == 'MEDIUM' else 'Continue monitoring current practices'} to ensure ethical AI deployment."""
        
        recommendations = [
            "Implement data rebalancing to ensure fair representation across all groups",
            "Apply fairness-aware machine learning techniques during model training",
            "Establish regular bias monitoring and alerting systems", 
            "Conduct legal review of AI decision-making processes",
            "Provide bias awareness training for technical and business teams"
        ]
        
        return {
            'ai_analysis': expert_analysis,
            'risk_level': risk_level,
            'bias_score': bias_score,
            'recommendations': recommendations,
            'legal_risks': [
                f"Potential violation of anti-discrimination laws in {use_case}",
                "Risk of regulatory penalties and legal challenges",
                "Possible class-action lawsuits from affected groups"
            ],
            'model_used': 'Expert Fallback Analysis',
            'timestamp': datetime.now().isoformat(),
            'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0}
        }
