import json
import uuid
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import sys

# Import OpenAI service
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from packages.shared_utils.src.openai_service import OpenAIService

class BiasReportChatbot:
    """Interactive chatbot for discussing bias analysis reports"""
    
    def __init__(self):
        try:
            self.openai_service = OpenAIService()
            self.chat_available = True
        except Exception as e:
            print(f"Chat service unavailable: {e}")
            self.chat_available = False
        
        self.conversation_history = {}
    
    def start_conversation(self, session_id: str, analysis_data: Dict) -> str:
        """Start a new conversation about a bias analysis report"""
        
        if not self.chat_available:
            return "Chat service is currently unavailable. Please refer to the PDF report for detailed analysis."
        
        # Initialize conversation history
        conversation_id = str(uuid.uuid4())
        self.conversation_history[conversation_id] = {
            'session_id': session_id,
            'analysis_data': analysis_data,
            'messages': [],
            'created_at': datetime.now().isoformat()
        }
        
        # Create initial context message
        context = self._create_report_context(analysis_data)
        
        welcome_message = f"""
        👋 Hi! I'm your AI Bias Analysis Assistant. I've reviewed your bias audit report and I'm here to help you understand the results.

        **Quick Summary:**
        • Risk Level: {analysis_data.get('ai_analysis', {}).get('risk_level', 'Unknown')}
        • Bias Score: {analysis_data.get('ai_analysis', {}).get('bias_score', 0):.3f}
        • Violations Found: {len(analysis_data.get('fairness_violations', []))}

        You can ask me questions like:
        • "Why is my system biased?"
        • "How can I fix this bias?"
        • "What do these numbers mean?"
        • "Is this legally compliant?"
        • "What should I do next?"

        What would you like to know about your bias analysis?
        """
        
        return {
            'conversation_id': conversation_id,
            'message': welcome_message,
            'suggestions': [
                "Why is my system showing bias?",
                "How serious is this bias level?", 
                "What steps should I take to fix this?",
                "Can you explain the technical metrics?",
                "What are the legal implications?"
            ]
        }
    
    def chat_with_report(self, conversation_id: str, user_message: str) -> Dict[str, Any]:
        """Handle user chat messages about the bias report"""
        
        if not self.chat_available:
            return {
                'error': 'Chat service unavailable',
                'message': 'Please refer to the PDF report for analysis details.'
            }
        
        if conversation_id not in self.conversation_history:
            return {
                'error': 'Conversation not found',
                'message': 'Please start a new conversation from the results page.'
            }
        
        conversation = self.conversation_history[conversation_id]
        analysis_data = conversation['analysis_data']
        
        # Add user message to history
        conversation['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        try:
            # Create context-aware prompt
            system_prompt = self._create_chat_system_prompt(analysis_data)
            
            # Prepare conversation for OpenAI
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add recent conversation history (last 10 messages)
            recent_messages = conversation['messages'][-10:]
            for msg in recent_messages:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            
            # Get AI response using GPT-5 nano
            import openai
            response = openai.chat.completions.create(
                model=self.openai_service.model,
                messages=messages,
                max_completion_tokens=1000,
                reasoning_effort="low",
                verbosity="medium"
            )
            
            ai_response = response.choices[0].message.content
            
            # Add AI response to history
            conversation['messages'].append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate follow-up suggestions
            suggestions = self._generate_followup_suggestions(user_message, analysis_data)
            
            return {
                'message': ai_response,
                'suggestions': suggestions,
                'conversation_id': conversation_id
            }
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error processing your question: {str(e)}"
            return {
                'message': error_message,
                'suggestions': ["Could you rephrase your question?", "Try asking about specific metrics"],
                'conversation_id': conversation_id
            }
    
    def _create_report_context(self, analysis_data: Dict) -> str:
        """Create context summary for the AI assistant"""
        
        dataset_info = analysis_data.get('dataset_info', {})
        ai_analysis = analysis_data.get('ai_analysis', {})
        violations = analysis_data.get('fairness_violations', [])
        
        context = f"""
        BIAS ANALYSIS REPORT CONTEXT:
        
        Dataset: {dataset_info.get('filename', 'Unknown')}
        Records: {dataset_info.get('total_records', 0)}
        Target Variable: {dataset_info.get('target_variable', 'Unknown')}
        Protected Attributes: {', '.join(dataset_info.get('protected_attributes', []))}
        
        BIAS ASSESSMENT:
        Risk Level: {ai_analysis.get('risk_level', 'Unknown')}
        Bias Score: {ai_analysis.get('bias_score', 0):.3f}
        Violations: {len(violations)}
        
        AI ANALYSIS SUMMARY:
        {ai_analysis.get('ai_analysis', 'No analysis available')[:500]}...
        """
        
        return context
    
    def _create_chat_system_prompt(self, analysis_data: Dict) -> str:
        """Create expert bias auditor system prompt for contextual chat"""
        
        context = self._create_report_context(analysis_data)
        
        system_prompt = f"""You are Bias Buster, an expert AI auditor and fairness consultant with deep expertise in AI ethics, bias detection, and regulatory compliance.

You are having a conversation with a business stakeholder about their bias analysis report. Your role is to:

CONTEXT:
{context}

EXPERTISE TO APPLY:
- Statistical bias interpretation (demographic parity, equal opportunity, calibration)
- Legal implications (anti-discrimination laws, GDPR, fair lending regulations)
- Business impact assessment and risk evaluation
- Practical remediation strategies and implementation guidance
- Intersectional bias analysis and real-world consequences

CONVERSATION GUIDELINES:
1. **Be the expert**: Draw on deep knowledge of bias patterns, legal precedents, and industry best practices
2. **Explain clearly**: Translate complex statistical concepts into business language
3. **Be actionable**: Provide specific, implementable recommendations with timelines
4. **Consider context**: Tailor advice to their specific use case and risk tolerance
5. **Be honest**: Don't downplay serious bias issues, but provide constructive guidance
6. **Think holistically**: Consider technical, legal, ethical, and business dimensions

RESPONSE STYLE:
- Professional consultant tone - authoritative yet accessible
- Use specific examples and analogies to explain concepts
- Provide concrete next steps with priorities and timelines
- Reference relevant laws, regulations, and industry standards when applicable
- Consider multiple stakeholder perspectives (technical, legal, business, affected communities)

CRITICAL: You are an expert consultant. Provide comprehensive, nuanced analysis that goes beyond surface-level observations. Help them understand not just what the numbers mean, but why they matter and what to do about them.
"""
        
        return system_prompt
    
    def _generate_followup_suggestions(self, user_message: str, analysis_data: Dict) -> List[str]:
        """Generate contextual follow-up questions"""
        
        risk_level = analysis_data.get('ai_analysis', {}).get('risk_level', 'UNKNOWN')
        violations = len(analysis_data.get('fairness_violations', []))
        
        # Context-aware suggestions
        if 'why' in user_message.lower() or 'what' in user_message.lower():
            suggestions = [
                "How can I fix this bias?",
                "What are the next steps?",
                "Is this legally compliant?"
            ]
        elif 'fix' in user_message.lower() or 'improve' in user_message.lower():
            suggestions = [
                "How long will it take to fix?",
                "What resources do I need?",
                "Can you prioritize the fixes?"
            ]
        elif 'legal' in user_message.lower() or 'compliance' in user_message.lower():
            suggestions = [
                "What documentation do I need?",
                "Should I consult legal counsel?",
                "Are there regulatory requirements?"
            ]
        else:
            # Default suggestions based on risk level
            if risk_level in ['CRITICAL', 'HIGH']:
                suggestions = [
                    "What's the most urgent fix needed?",
                    "How serious are the legal risks?",
                    "Can I deploy this system safely?"
                ]
            else:
                suggestions = [
                    "How can I improve fairness further?",
                    "Should I run regular audits?",
                    "What metrics should I monitor?"
                ]
        
        return suggestions

# Global instance
bias_chatbot = BiasReportChatbot()
