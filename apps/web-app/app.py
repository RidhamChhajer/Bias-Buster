from flask import Flask, request, render_template, redirect, url_for, flash, send_file, jsonify
import os
import json
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
import sys

# Add project root to path for imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import our custom modules
from packages.bias_detector.src.bias_detector import BiasDetector
from packages.report_generator.src.pdf_generator import PDFReportGenerator
from packages.shared_utils.src.openai_service import OpenAIService

app = Flask(__name__)
app.secret_key = 'bias-buster-secret-key-2025'

# Configuration
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['REPORTS_FOLDER'] = Path(__file__).parent / 'reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['REPORTS_FOLDER'].mkdir(exist_ok=True)

# Initialize services
services = {}
try:
    services['openai'] = OpenAIService()
    print("✅ OpenAI GPT-5 nano service initialized")
except Exception as e:
    print(f"⚠️ OpenAI service unavailable: {e}")
    services['openai'] = None

services['bias_detector'] = BiasDetector()
services['pdf_generator'] = PDFReportGenerator()
print("✅ Bias detection and PDF generation services initialized")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and bias analysis"""
    
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash('Invalid file format. Please upload CSV, Excel, or JSON files.', 'error')
        return redirect(request.url)
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        session_id = str(uuid.uuid4())
        file_path = app.config['UPLOAD_FOLDER'] / f"{session_id}_{filename}"
        file.save(file_path)
        
        # Get form data
        protected_attributes = request.form.getlist('protected_attributes')
        custom_attributes = request.form.get('custom_attributes', '').strip()
        target_variable = request.form.get('target_variable', '').strip()
        
        # Add custom attributes
        if custom_attributes:
            custom_attrs = [attr.strip() for attr in custom_attributes.split(',')]
            protected_attributes.extend(custom_attrs)
        
        if not protected_attributes or not target_variable:
            flash('Please select protected attributes and target variable', 'error')
            return redirect(url_for('index'))
        
        # Load and analyze dataset
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif filename.endswith('.json'):
                df = pd.read_json(file_path)
        except Exception as e:
            flash(f'Error reading file: {str(e)}', 'error')
            return redirect(url_for('index'))
        
        # Validate columns exist
        missing_cols = []
        if target_variable not in df.columns:
            missing_cols.append(target_variable)
        
        for attr in protected_attributes:
            if attr not in df.columns:
                missing_cols.append(attr)
        
        if missing_cols:
            flash(f'Columns not found in dataset: {", ".join(missing_cols)}', 'error')
            return redirect(url_for('index'))
        
        # Perform bias analysis
        print(f"🔍 Analyzing bias for: {protected_attributes} -> {target_variable}")
        
        bias_results = services['bias_detector'].analyze_bias(
            df, protected_attributes, target_variable
        )
        
        # Dataset information
        dataset_info = {
            'filename': filename,
            'total_records': len(df),
            'columns': list(df.columns),
            'protected_attributes': protected_attributes,
            'target_variable': target_variable
        }
        
        # AI Analysis using enhanced prompt
        ai_analysis = {}
        if services['openai']:
            try:
                print("🤖 Getting GPT-5 nano expert analysis...")
                ai_analysis = services['openai'].analyze_bias_with_context(
                    bias_results, {'dataset_info': dataset_info}
                )
            except Exception as e:
                print(f"⚠️ AI analysis failed: {e}")
                ai_analysis = generate_fallback_ai_analysis(bias_results, dataset_info)
        else:
            ai_analysis = generate_fallback_ai_analysis(bias_results, dataset_info)
        
        # Create comprehensive results
        analysis_results = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': dataset_info,
            'bias_metrics': bias_results['bias_metrics'],
            'fairness_violations': bias_results['fairness_violations'],
            'summary': {
                'total_records': len(df),
                'protected_attributes': protected_attributes,
                'target_variable': target_variable,
                'violations_found': len(bias_results['fairness_violations'])
            },
            'ai_analysis': ai_analysis
        }
        
        # Save analysis results
        results_file = app.config['REPORTS_FOLDER'] / f'analysis_{session_id}.json'
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print("✅ Analysis complete!")
        
        # Redirect to results page
        return render_template('results.html', 
                             results=analysis_results, 
                             session_id=session_id)
        
    except Exception as e:
        flash(f'Analysis failed: {str(e)}', 'error')
        print(f"❌ Analysis error: {e}")
        return redirect(url_for('index'))

@app.route('/download_report/<session_id>')
def download_report(session_id):
    """Generate and download PDF report"""
    
    try:
        # Load analysis results
        results_file = app.config['REPORTS_FOLDER'] / f'analysis_{session_id}.json'
        
        if not results_file.exists():
            flash('Report not found', 'error')
            return redirect(url_for('index'))
        
        with open(results_file, 'r') as f:
            analysis_data = json.load(f)
        
        # Generate PDF report
        pdf_filename = f'bias_audit_report_{session_id}.pdf'
        pdf_path = app.config['REPORTS_FOLDER'] / pdf_filename
        
        success = services['pdf_generator'].generate_bias_report(analysis_data, pdf_path)
        
        if success and pdf_path.exists():
            return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
        else:
            flash('Failed to generate PDF report', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Report generation failed: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/start_chat', methods=['POST'])
def start_chat():
    """Start a new chat conversation about bias analysis results"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Load analysis results
        results_file = app.config['REPORTS_FOLDER'] / f'analysis_{session_id}.json'
        
        if not results_file.exists():
            return jsonify({'error': 'Analysis results not found'}), 404
        
        with open(results_file, 'r') as f:
            analysis_data = json.load(f)
        
        # Try to use chat service if available
        if services.get('openai'):
            try:
                # Import chat service
                from chat_interface import bias_chatbot
                
                chat_response = bias_chatbot.start_conversation(session_id, analysis_data)
                return jsonify(chat_response)
                
            except Exception as e:
                print(f"Chat service error: {e}")
                # Fall back to basic response
                pass
        
        # Fallback chat response
        risk_level = analysis_data.get('ai_analysis', {}).get('risk_level', 'Unknown')
        bias_score = analysis_data.get('ai_analysis', {}).get('bias_score', 0)
        violations = len(analysis_data.get('fairness_violations', []))
        
        fallback_message = f"""
        👋 Hi! I'm your Bias Analysis Assistant. I've reviewed your analysis results.

        **Quick Summary:**
        • Risk Level: {risk_level}
        • Bias Score: {bias_score:.3f}
        • Issues Found: {violations}

        While I can't provide interactive chat right now (OpenAI service unavailable), 
        I can tell you that your results show {"significant bias concerns" if risk_level in ['CRITICAL', 'HIGH'] else "some bias patterns" if risk_level == 'MEDIUM' else "minimal bias issues"}.

        **What you should do:**
        {"🚨 Take immediate action to address the bias before deploying your system" if risk_level == 'CRITICAL' else
         "⚠️ Implement bias mitigation strategies" if risk_level == 'HIGH' else  
         "✅ Monitor and make minor improvements" if risk_level in ['MEDIUM', 'LOW'] else
         "✅ Continue current practices"}

        Please download the detailed PDF report for comprehensive recommendations.
        """
        
        return jsonify({
            'conversation_id': f"fallback_{session_id}",
            'message': fallback_message,
            'suggestions': [
                "Download the detailed PDF report",
                "What does my bias score mean?",
                "How serious is this risk level?",
                "What should I do next?"
            ]
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start chat: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat_message():
    """Handle chat messages"""
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        user_message = data.get('message')
        
        if not conversation_id or not user_message:
            return jsonify({'error': 'Conversation ID and message required'}), 400
        
        # Check if it's a fallback conversation
        if conversation_id.startswith('fallback_'):
            session_id = conversation_id.replace('fallback_', '')
            
            # Load analysis data for context
            results_file = app.config['REPORTS_FOLDER'] / f'analysis_{session_id}.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # Generate contextual responses
                response = generate_fallback_chat_response(user_message, analysis_data)
                return jsonify(response)
        
        # Try to use full chat service if available
        if services.get('openai'):
            try:
                from chat_interface import bias_chatbot
                
                chat_response = bias_chatbot.chat_with_report(conversation_id, user_message)
                return jsonify(chat_response)
                
            except Exception as e:
                print(f"Chat service error: {e}")
                pass
        
        # Fallback response
        return jsonify({
            'message': "I apologize, but I'm not able to provide detailed responses right now. Please refer to your PDF report for comprehensive analysis and recommendations.",
            'suggestions': [
                "Download the PDF report",
                "Try uploading a new dataset",
                "Check the detailed analysis section"
            ],
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'}), 500

def generate_fallback_ai_analysis(bias_results: dict, dataset_info: dict) -> dict:
    """Generate fallback AI analysis when OpenAI service is unavailable"""
    
    violations = bias_results.get('fairness_violations', [])
    bias_metrics = bias_results.get('bias_metrics', {})
    
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
    
    # Determine risk level
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
    target_var = dataset_info.get('target_variable', 'outcomes')
    protected_attrs = ', '.join(dataset_info.get('protected_attributes', []))
    
    ai_analysis = f"""**🎯 BIAS VERDICT**: {'UNFAIR' if risk_level in ['CRITICAL', 'HIGH', 'MEDIUM'] else 'MOSTLY FAIR'} - {risk_level} Risk Level

As an expert AI auditor, I've identified {"significant" if risk_level in ['CRITICAL', 'HIGH'] else "moderate" if risk_level == 'MEDIUM' else "minimal"} bias concerns in your system.

**📊 EVIDENCE**: 
- {len(violations)} fairness violations detected across {protected_attrs}
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
{risk_level.title()} bias in decision systems can result in discriminatory outcomes, legal violations, and loss of stakeholder trust. {'Immediate action is required' if risk_level in ['CRITICAL', 'HIGH'] else 'Proactive improvements recommended' if risk_level == 'MEDIUM' else 'Continue monitoring current practices'} to ensure ethical AI deployment."""
    
    recommendations = [
        "Implement data rebalancing to ensure fair representation across all groups",
        "Apply fairness-aware machine learning techniques during model training",
        "Establish regular bias monitoring and alerting systems", 
        "Conduct legal review of AI decision-making processes",
        "Provide bias awareness training for technical and business teams"
    ]
    
    return {
        'ai_analysis': ai_analysis,
        'risk_level': risk_level,
        'bias_score': bias_score,
        'recommendations': recommendations,
        'model_used': 'Expert Fallback Analysis',
        'timestamp': datetime.now().isoformat(),
        'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0}
    }

def generate_fallback_chat_response(user_message: str, analysis_data: dict) -> dict:
    """Generate basic chat responses when full AI chat is unavailable"""
    
    risk_level = analysis_data.get('ai_analysis', {}).get('risk_level', 'Unknown')
    bias_score = analysis_data.get('ai_analysis', {}).get('bias_score', 0)
    violations = analysis_data.get('fairness_violations', [])
    
    message_lower = user_message.lower()
    
    # Simple keyword-based responses
    if any(word in message_lower for word in ['bias score', 'score mean', 'what does']):
        response = f"""
        Your bias score of {bias_score:.3f} means:
        
        • **0.0-0.2**: Minimal bias (good!)
        • **0.2-0.4**: Low bias (monitor)
        • **0.4-0.6**: Medium bias (improve)
        • **0.6-0.8**: High bias (fix needed)
        • **0.8-1.0**: Critical bias (urgent action)
        
        Your score of {bias_score:.3f} indicates {"minimal bias" if bias_score < 0.2 else "low bias" if bias_score < 0.4 else "medium bias" if bias_score < 0.6 else "high bias" if bias_score < 0.8 else "critical bias"}.
        """
        suggestions = ["How can I improve this score?", "What causes bias?", "Is this legally compliant?"]
        
    elif any(word in message_lower for word in ['risk level', 'serious', 'how bad']):
        risk_explanations = {
            'CRITICAL': "This is the highest risk level. Your system shows severe discrimination that could lead to lawsuits and serious harm. Stop using it immediately.",
            'HIGH': "This is a serious risk level. Your system shows clear unfair treatment that needs immediate attention.",
            'MEDIUM': "This is a moderate risk level. Your system has concerning patterns that should be fixed.",
            'LOW': "This is a low risk level. Your system has minor issues worth monitoring.",
            'MINIMAL': "This is the best outcome. Your system appears fair with minimal issues."
        }
        
        response = f"""
        Your **{risk_level}** risk level means:
        
        {risk_explanations.get(risk_level, 'Unable to determine risk level explanation.')}
        
        You have **{len(violations)} fairness violations** detected.
        """
        suggestions = ["What should I do next?", "How do I fix this?", "What are the violations?"]
        
    elif any(word in message_lower for word in ['fix', 'improve', 'what do', 'next']):
        if risk_level in ['CRITICAL', 'HIGH']:
            response = """
            **Immediate Actions Needed:**
            
            1. **Stop deployment** - Don't use this system for real decisions
            2. **Review your data** - Check for biased training data
            3. **Consult experts** - Get legal and technical advice
            4. **Implement fixes** - Rebalance data, adjust algorithms
            5. **Test again** - Re-run bias analysis after changes
            
            This is urgent - bias at this level can cause serious harm and legal issues.
            """
        else:
            response = """
            **Recommended Actions:**
            
            1. **Monitor regularly** - Set up ongoing bias checks
            2. **Improve data quality** - Ensure representative samples
            3. **Document findings** - Keep records for compliance
            4. **Train your team** - Educate on bias awareness
            5. **Regular audits** - Check bias periodically
            
            Your system is relatively fair but can always be improved.
            """
        suggestions = ["How long will fixes take?", "Do I need legal help?", "Can I use this system now?"]
        
    elif any(word in message_lower for word in ['legal', 'compliance', 'lawsuit', 'law']):
        response = f"""
        **Legal Implications:**
        
        With a **{risk_level}** risk level:
        
        • **Regulatory risk**: {"High - may violate anti-discrimination laws" if risk_level in ['CRITICAL', 'HIGH'] else "Medium - monitor compliance" if risk_level == 'MEDIUM' else "Low - likely compliant"}
        • **Lawsuit risk**: {"High - discrimination claims possible" if risk_level in ['CRITICAL', 'HIGH'] else "Medium - some risk exists" if risk_level == 'MEDIUM' else "Low - minimal legal risk"}
        • **Documentation**: {"Essential - you need legal review" if risk_level in ['CRITICAL', 'HIGH'] else "Recommended - keep bias audit records"}
        
        {"⚠️ Consider consulting with legal counsel immediately." if risk_level in ['CRITICAL', 'HIGH'] else "📋 Maintain good documentation for compliance."}
        """
        suggestions = ["Should I stop using my system?", "What laws apply?", "Do I need a lawyer?"]
        
    else:
        # Generic response
        response = f"""
        I'd like to help you understand your bias analysis better. 
        
        **Your Results Summary:**
        • Risk Level: {risk_level}
        • Bias Score: {bias_score:.3f}/1.0
        • Issues Found: {len(violations)}
        
        For detailed technical analysis and recommendations, please download your PDF report. 
        It contains comprehensive insights and step-by-step guidance.
        """
        suggestions = ["What does my bias score mean?", "How serious is my risk level?", "What should I do next?"]
    
    return {
        'message': response.strip(),
        'suggestions': suggestions,
        'conversation_id': f"fallback_{analysis_data.get('session_id', 'unknown')}"
    }

if __name__ == '__main__':
    print("🚀 Starting Bias Buster Platform...")
    print("📊 Access the platform at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
