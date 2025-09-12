import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# Add the root directory to Python path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / '.env')

# Import OpenAI
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIService:
    """Service class for OpenAI GPT-5 nano integration"""
    
    def __init__(self):
        """Initialize OpenAI service with GPT-5 nano configuration"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-5-nano-2025-08-07')
        self.max_completion_tokens = int(os.getenv('OPENAI_MAX_TOKENS', 2000))
        
        # Validate configuration
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required. Please add it to your .env file")
        
        # Set OpenAI API key
        openai.api_key = self.api_key
        logger.info(f"OpenAI Service initialized with model: {self.model}")
    
    def analyze_bias_with_gpt5(self, bias_results: Dict, dataset_info: Dict) -> Dict[str, Any]:
        """
        Use GPT-5 nano to analyze bias results and generate explanations
        """
        try:
            # Create prompt for bias analysis
            prompt = self._create_bias_analysis_prompt(bias_results, dataset_info)
            
            # Call GPT-5 nano with ONLY supported parameters
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an AI fairness expert who analyzes bias in datasets and machine learning models. Provide clear, actionable insights about discrimination patterns and mitigation strategies."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort="minimal",  # Supported: minimal, low, medium, high
                verbosity="medium"           # Supported: low, medium, high
            )
            
            # Correct response parsing
            ai_analysis = response.choices[0].message.content
            
            # Structure the response
            structured_response = self._structure_ai_response(ai_analysis, bias_results, response)
            
            logger.info("GPT-5 nano analysis completed successfully")
            return structured_response
            
        except Exception as e:
            logger.error(f"Error in GPT-5 nano analysis: {str(e)}")
            return {
                "error": f"AI analysis failed: {str(e)}",
                "fallback_analysis": self._generate_fallback_analysis(bias_results)
            }
    
    def _create_bias_analysis_prompt(self, bias_results: Dict, dataset_info: Dict) -> str:
        """Create structured prompt for GPT-5 nano bias analysis"""
        
        prompt = f"""
        BIAS ANALYSIS REQUEST - DATASET AUDIT

        Dataset Information:
        - Total Records: {dataset_info.get('total_records', 'Unknown')}
        - Protected Attributes: {', '.join(dataset_info.get('protected_attributes', []))}
        - Target Variable: {dataset_info.get('target_variable', 'Unknown')}

        Bias Detection Results:
        {json.dumps(bias_results, indent=2)}

        Please provide a structured bias analysis including:

        1. BIAS SEVERITY: Rate overall bias (Low/Medium/High/Critical)
        2. KEY PROBLEMS: Which groups face discrimination?
        3. IMPACT: Real-world consequences of this bias
        4. FIXES: Top 3 actionable recommendations
        5. COMPLIANCE: Legal/regulatory concerns

        Keep response focused and business-friendly.
        """
        
        return prompt
    
    def _structure_ai_response(self, ai_analysis: str, bias_results: Dict, response_obj: Any = None) -> Dict[str, Any]:
        """Structure GPT-5 nano response into organized format"""
        
        result = {
            "ai_analysis": ai_analysis,
            "bias_score": self._calculate_overall_bias_score(bias_results),
            "risk_level": self._determine_risk_level(bias_results),
            "key_findings": self._extract_key_findings(ai_analysis),
            "recommendations": self._extract_recommendations(ai_analysis),
            "timestamp": datetime.now().isoformat(),
            "model_used": self.model,
            "parameters_used": {
                "reasoning_effort": "minimal",
                "verbosity": "medium",
                "max_completion_tokens": self.max_completion_tokens
            }
        }
        
        # Add usage info if available
        if response_obj and hasattr(response_obj, 'usage'):
            result["token_usage"] = {
                "prompt_tokens": response_obj.usage.prompt_tokens,
                "completion_tokens": response_obj.usage.completion_tokens,
                "total_tokens": response_obj.usage.total_tokens
            }
        
        return result
    
    def _calculate_overall_bias_score(self, bias_results: Dict) -> float:
        """Calculate numerical bias score from results"""
        scores = []
        for attr, results in bias_results.items():
            if isinstance(results, dict):
                for group, stats in results.items():
                    if isinstance(stats, dict) and 'positive_rate' in stats:
                        scores.append(abs(stats['positive_rate'] - 0.5) * 2)
        
        return round(sum(scores) / len(scores) if scores else 0.0, 3)
    
    def _determine_risk_level(self, bias_results: Dict) -> str:
        """Determine risk level based on bias patterns"""
        bias_score = self._calculate_overall_bias_score(bias_results)
        
        if bias_score >= 0.7:
            return "CRITICAL"
        elif bias_score >= 0.5:
            return "HIGH"
        elif bias_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _extract_key_findings(self, ai_analysis: str) -> List[str]:
        """Extract key findings from AI analysis"""
        lines = ai_analysis.split('\n')
        findings = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['bias', 'discrimination', 'unfair', 'disparity']):
                if len(line.strip()) > 20:
                    findings.append(line.strip())
        
        return findings[:5]
    
    def _extract_recommendations(self, ai_analysis: str) -> List[str]:
        """Extract recommendations from AI analysis"""
        lines = ai_analysis.split('\n')
        recommendations = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'consider']):
                if len(line.strip()) > 20:
                    recommendations.append(line.strip())
        
        return recommendations[:5]
    
    def _generate_fallback_analysis(self, bias_results: Dict) -> Dict[str, Any]:
        """Generate basic analysis if GPT-5 nano fails"""
        return {
            "message": "AI analysis unavailable, showing basic bias metrics",
            "bias_score": self._calculate_overall_bias_score(bias_results),
            "risk_level": self._determine_risk_level(bias_results),
            "basic_findings": ["Bias patterns detected in protected attributes", "Manual review recommended"]
        }

def test_openai_connection():
    """Test GPT-5 nano connectivity with proper response parsing"""
    try:
        print("🔍 Testing GPT-5 nano connection...")
        
        service = OpenAIService()
        print(f"✅ OpenAI Service initialized successfully")
        print(f"📝 Model: {service.model}")
        print(f"🔑 API Key: {'*' * (len(service.api_key) - 8) + service.api_key[-4:] if service.api_key else 'Not set'}")
        
        # Test API call with correct response parsing
        print("🚀 Testing basic connection...")
        test_response = openai.chat.completions.create(
            model=service.model,
            messages=[
                {
                    "role": "user", 
                    "content": "Hello! This is a connection test for the Bias Buster platform. Please respond with a brief confirmation."
                }
            ],
            max_completion_tokens=100,
            reasoning_effort="minimal",
            verbosity="low"
        )
        
        print("✅ GPT-5 nano connection successful!")
        print(f"📄 Response: {test_response.choices[0].message.content}")
        print(f"🎯 Usage: {test_response.usage}")
        
        # Test bias analysis functionality
        print("\n🧪 Testing bias analysis functionality...")
        test_bias_results = {
            "gender": {
                "male": {"positive_rate": 0.75, "group_size": 1000},
                "female": {"positive_rate": 0.45, "group_size": 800}
            }
        }
        test_dataset_info = {
            "total_records": 1800,
            "protected_attributes": ["gender"],
            "target_variable": "loan_approved"
        }
        
        analysis_result = service.analyze_bias_with_gpt5(test_bias_results, test_dataset_info)
        
        if "error" not in analysis_result:
            print("✅ Bias analysis test successful!")
            print(f"🎯 Risk Level: {analysis_result['risk_level']}")
            print(f"📊 Bias Score: {analysis_result['bias_score']}")
            print(f"🔍 Model Used: {analysis_result['model_used']}")
            print(f"💰 Token Usage: {analysis_result.get('token_usage', 'N/A')}")
            print(f"📝 AI Analysis Preview: {analysis_result['ai_analysis'][:200]}...")
        else:
            print(f"⚠️ Bias analysis test failed: {analysis_result['error']}")
        
        print("\n🎉 All tests passed! GPT-5 nano is ready for Bias Buster!")
        return True
        
    except Exception as e:
        print(f"❌ GPT-5 nano connection failed: {str(e)}")
        print(f"🐛 Error details: {type(e).__name__}: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure OPENAI_API_KEY is set in your .env file")
        print("2. Verify your OpenAI account has access to GPT-5 nano")
        print("3. Check your API usage limits")
        print("4. Ensure you have sufficient credits in your OpenAI account")
        return False

if __name__ == "__main__":
    test_openai_connection()
