import os
import sys
from pathlib import Path

# Add the root directory to Python path FIRST
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

print(f"🔍 Root directory: {ROOT_DIR}")
print(f"🔍 Python path: {sys.path[:3]}")

# Now try imports
try:
    from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
    from werkzeug.utils import secure_filename
    import pandas as pd
    import numpy as np
    import json
    from datetime import datetime
    import uuid
    print("✅ Basic imports successful")
except ImportError as e:
    print(f"❌ Basic import error: {e}")
    sys.exit(1)

# Try our custom imports with fallbacks
try:
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / '.env')
    print("✅ Environment loaded")
    
    # Import OpenAI service
    try:
        from packages.shared_utils.src.openai_service import OpenAIService
        print("✅ OpenAI service imported")
        openai_available = True
    except ImportError as e:
        print(f"⚠️ OpenAI service import failed: {e}")
        openai_available = False
        
    # Import bias detector - create minimal version if import fails
    try:
        from packages.bias_detector.src.custom_bias_detector import CustomBiasDetector  
        print("✅ Bias detector imported")
        bias_detector_available = True
    except ImportError as e:
        print(f"⚠️ Bias detector import failed: {e}")
        bias_detector_available = False
        
    # Import PDF generator - create minimal version if import fails  
    try:
        from packages.report_generator.src.pdf_generator import PDFReportGenerator
        print("✅ PDF generator imported") 
        pdf_generator_available = True
    except ImportError as e:
        print(f"⚠️ PDF generator import failed: {e}")
        pdf_generator_available = False
        
except Exception as e:
    print(f"❌ Custom imports failed: {e}")
    print("🔧 Will use fallback implementations")
    openai_available = False
    bias_detector_available = False  
    pdf_generator_available = False

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'bias-buster-secret-key-2025'
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['REPORTS_FOLDER'] = Path(__file__).parent / 'reports' 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}

# Ensure upload/reports directories exist
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['REPORTS_FOLDER'].mkdir(exist_ok=True)

# Initialize services with fallbacks
services = {}

if openai_available:
    try:
        services['openai'] = OpenAIService()
        print("✅ OpenAI service initialized")
    except Exception as e:
        print(f"⚠️ OpenAI service init failed: {e}")
        services['openai'] = None
else:
    services['openai'] = None

if bias_detector_available:
    try:
        services['bias_detector'] = CustomBiasDetector()
        print("✅ Bias detector initialized")
    except Exception as e:
        print(f"⚠️ Bias detector init failed: {e}")
        services['bias_detector'] = None
else:
    services['bias_detector'] = None

if pdf_generator_available:
    try:
        services['pdf_generator'] = PDFReportGenerator()
        print("✅ PDF generator initialized")
    except Exception as e:
        print(f"⚠️ PDF generator init failed: {e}")
        services['pdf_generator'] = None
else:
    services['pdf_generator'] = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fallback bias detection function
def simple_bias_analysis(data, target_col, protected_attrs):
    """Simple bias analysis fallback"""
    results = {
        'summary': {
            'total_records': len(data),
            'target_variable': target_col,
            'protected_attributes': protected_attrs,
            'analysis_timestamp': datetime.now().isoformat()
        },
        'bias_metrics': {},
        'overall_bias_score': 0.0,
        'fairness_violations': [],
        'risk_assessment': {
            'level': 'LOW',
            'score': 0.0,
            'recommendation': 'Basic analysis completed - full analysis requires custom modules'
        }
    }
    
    for attr in protected_attrs:
        if attr in data.columns:
            groups = data[attr].unique()
            group_stats = {}
            
            for group in groups:
                group_data = data[data[attr] == group]
                group_size = len(group_data)
                
                # Calculate positive rate
                if target_col in data.columns:
                    if data[target_col].dtype in ['int64', 'float64', 'bool']:
                        positive_rate = group_data[target_col].mean()
                    else:
                        # For categorical targets, use most common value
                        mode_val = data[target_col].mode()[0]
                        positive_rate = (group_data[target_col] == mode_val).mean()
                else:
                    positive_rate = 0.5
                
                group_stats[str(group)] = {
                    'group_size': int(group_size),
                    'positive_rate': float(positive_rate),
                    'representation': float(group_size / len(data))
                }
            
            # Calculate simple disparities
            rates = [stats['positive_rate'] for stats in group_stats.values()]
            if len(rates) > 1:
                disparity = max(rates) - min(rates)
                results['bias_metrics'][attr] = {
                    'groups': group_stats,
                    'disparities': {
                        'statistical_parity_difference': disparity,
                        'demographic_parity_ratio': min(rates) / max(rates) if max(rates) > 0 else 0
                    },
                    'fairness_metrics': {
                        'passes_80_percent_rule': (min(rates) / max(rates)) >= 0.8 if max(rates) > 0 else True
                    }
                }
                
                # Update overall bias score
                results['overall_bias_score'] = max(results['overall_bias_score'], disparity)
    
    return results

@app.route('/')
def index():
    """Home page with file upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process bias detection"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Secure filename and save
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            filepath = app.config['UPLOAD_FOLDER'] / unique_filename
            file.save(filepath)
            
            # Get form data
            protected_attributes = request.form.getlist('protected_attributes')
            target_variable = request.form.get('target_variable')
            
            # Add custom attributes if provided
            custom_attrs = request.form.get('custom_attributes', '').strip()
            if custom_attrs:
                custom_list = [attr.strip() for attr in custom_attrs.split(',') if attr.strip()]
                protected_attributes.extend(custom_list)
            
            if not protected_attributes or not target_variable:
                flash('Please specify protected attributes and target variable', 'error')
                return redirect(url_for('index'))
            
            # Process the file
            analysis_result = process_uploaded_file(
                filepath, 
                protected_attributes, 
                target_variable,
                unique_filename
            )
            
            if analysis_result['success']:
                flash('Analysis completed successfully!', 'success')
                return redirect(url_for('results', session_id=analysis_result['session_id']))
            else:
                flash(f'Error processing file: {analysis_result["error"]}', 'error')
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'Error uploading file: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload CSV, XLSX, or JSON files only.', 'error')
        return redirect(url_for('index'))

def process_uploaded_file(filepath, protected_attributes, target_variable, filename):
    """Process uploaded file and run bias detection"""
    try:
        print(f"🔍 Processing file: {filepath}")
        
        # Load data based on file type
        file_ext = filepath.suffix.lower()
        
        if file_ext == '.csv':
            data = pd.read_csv(filepath)
        elif file_ext in ['.xlsx', '.xls']:
            data = pd.read_excel(filepath)
        elif file_ext == '.json':
            data = pd.read_json(filepath)
        else:
            return {'success': False, 'error': 'Unsupported file type'}
        
        print(f"✅ Data loaded: {len(data)} rows, {len(data.columns)} columns")
        
        # Validate data
        if data.empty:
            return {'success': False, 'error': 'File is empty'}
        
        # Check if specified columns exist
        missing_cols = []
        for attr in protected_attributes:
            if attr not in data.columns:
                missing_cols.append(attr)
        
        if target_variable not in data.columns:
            missing_cols.append(target_variable)
        
        if missing_cols:
            available_cols = list(data.columns)
            return {
                'success': False, 
                'error': f'Missing columns: {", ".join(missing_cols)}. Available columns: {", ".join(available_cols)}'
            }
        
        # Run bias detection (use available service or fallback)
        if services['bias_detector']:
            print("🔍 Using custom bias detector...")
            bias_results = services['bias_detector'].detect_bias(data, target_variable, protected_attributes)
        else:
            print("🔍 Using fallback bias analysis...")
            bias_results = simple_bias_analysis(data, target_variable, protected_attributes)
        
        # Prepare dataset info
        dataset_info = {
            'filename': filename,
            'total_records': len(data),
            'protected_attributes': protected_attributes,
            'target_variable': target_variable,
            'columns': list(data.columns),
            'data_types': {str(k): str(v) for k, v in data.dtypes.to_dict().items()}
        }
        
        # Get AI analysis (use available service or fallback)
        if services['openai']:
            print("🤖 Getting GPT-5 nano analysis...")
            try:
                ai_analysis = services['openai'].analyze_bias_with_gpt5(bias_results, dataset_info)
            except Exception as e:
                print(f"⚠️ AI analysis failed: {e}")
                ai_analysis = create_fallback_ai_analysis(bias_results)
        else:
            print("🤖 Using fallback AI analysis...")
            ai_analysis = create_fallback_ai_analysis(bias_results)
        
        # Generate session ID and save results
        session_id = str(uuid.uuid4())
        
        # Save analysis results
        results_data = {
            'session_id': session_id,
            'filename': filename,
            'dataset_info': dataset_info,
            'summary': bias_results.get('summary', {}),
            'bias_metrics': bias_results.get('bias_metrics', {}),
            'fairness_violations': bias_results.get('fairness_violations', []),
            'ai_analysis': ai_analysis,
            'timestamp': datetime.now().isoformat(),
            'file_path': str(filepath)
        }
        
        # Save to JSON file for later retrieval
        results_file = app.config['REPORTS_FOLDER'] / f'analysis_{session_id}.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"✅ Analysis complete, saved to: {results_file}")
        return {'success': True, 'session_id': session_id}
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def create_fallback_ai_analysis(bias_results):
    """Create fallback AI analysis when GPT-5 nano is not available"""
    overall_score = bias_results.get('overall_bias_score', 0.0)
    violations = bias_results.get('fairness_violations', [])
    
    # Determine risk level
    if overall_score >= 0.7 or len(violations) >= 3:
        risk_level = 'CRITICAL'
    elif overall_score >= 0.5 or len(violations) >= 2:
        risk_level = 'HIGH'
    elif overall_score >= 0.3 or len(violations) >= 1:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    analysis_text = f"""
    BIAS ANALYSIS SUMMARY
    
    Overall Bias Assessment: This dataset shows {risk_level.lower()} levels of potential bias based on statistical analysis.
    
    Key Findings:
    - Overall bias score: {overall_score:.3f} (0 = no bias, 1 = high bias)
    - Number of fairness violations detected: {len(violations)}
    - Risk level assessment: {risk_level}
    
    Statistical Analysis:
    The analysis examined disparities in outcomes across different demographic groups. 
    {'Significant disparities were detected' if overall_score > 0.3 else 'No major disparities were detected'} 
    in the protected attributes analyzed.
    
    Recommendations:
    {'Immediate bias mitigation measures are recommended' if risk_level in ['CRITICAL', 'HIGH'] else 'Continue monitoring for bias patterns'}.
    Consider implementing fairness-aware algorithms and regular bias auditing.
    
    Note: This analysis was performed using basic statistical methods. 
    For enhanced AI-powered insights, ensure GPT-5 nano integration is properly configured.
    """
    
    return {
        'ai_analysis': analysis_text,
        'bias_score': overall_score,
        'risk_level': risk_level,
        'key_findings': [
            f"Overall bias score: {overall_score:.3f}",
            f"Risk level: {risk_level}",
            f"Violations detected: {len(violations)}"
        ],
        'recommendations': [
            "Implement bias monitoring systems",
            "Consider fairness-aware machine learning approaches",  
            "Regular auditing of model decisions"
        ],
        'model_used': 'Statistical Analysis (Fallback)',
        'timestamp': datetime.now().isoformat()
    }

@app.route('/results/<session_id>')
def results(session_id):
    """Display bias detection results"""
    try:
        # Load analysis results
        results_file = app.config['REPORTS_FOLDER'] / f'analysis_{session_id}.json'
        
        if not results_file.exists():
            flash('Analysis results not found', 'error')
            return redirect(url_for('index'))
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        return render_template('results.html', 
                             results=results_data,
                             session_id=session_id)
        
    except Exception as e:
        flash(f'Error loading results: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download_report/<session_id>')
def download_report(session_id):
    """Generate and download PDF report"""
    try:
        # Load analysis results
        results_file = app.config['REPORTS_FOLDER'] / f'analysis_{session_id}.json'
        
        if not results_file.exists():
            flash('Analysis results not found', 'error')
            return redirect(url_for('index'))
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Generate PDF report (use available service or create simple text file)
        if services['pdf_generator']:
            print("📄 Generating PDF report...")
            pdf_filename = f'bias_audit_report_{session_id}.pdf'
            pdf_path = app.config['REPORTS_FOLDER'] / pdf_filename
            
            services['pdf_generator'].generate_bias_report(results_data, pdf_path)
            
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=f'BiasAuditReport_{results_data["filename"]}_{datetime.now().strftime("%Y%m%d")}.pdf',
                mimetype='application/pdf'
            )
        else:
            # Fallback: create simple text report
            print("📄 Generating text report fallback...")
            report_content = create_text_report(results_data)
            
            report_filename = f'bias_audit_report_{session_id}.txt'
            report_path = app.config['REPORTS_FOLDER'] / report_filename
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            return send_file(
                report_path,
                as_attachment=True,
                download_name=f'BiasAuditReport_{results_data["filename"]}_{datetime.now().strftime("%Y%m%d")}.txt',
                mimetype='text/plain'
            )
        
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('results', session_id=session_id))

def create_text_report(results_data):
    """Create a simple text report as fallback"""
    report = f"""
BIAS AUDIT REPORT
================

Dataset: {results_data['dataset_info']['filename']}
Generated: {results_data['timestamp']}
Total Records: {results_data['dataset_info']['total_records']:,}

RISK ASSESSMENT
===============
Risk Level: {results_data['ai_analysis']['risk_level']}
Bias Score: {results_data['ai_analysis']['bias_score']:.3f}

ANALYSIS SUMMARY
===============
{results_data['ai_analysis']['ai_analysis']}

DETAILED FINDINGS
================
"""
    
    for attr, metrics in results_data.get('bias_metrics', {}).items():
        report += f"\n{attr.upper()} ANALYSIS:\n"
        report += "-" * 30 + "\n"
        
        for group, stats in metrics.get('groups', {}).items():
            report += f"{group}: {stats['positive_rate']:.1%} positive rate ({stats['group_size']} samples)\n"
        
        disparities = metrics.get('disparities', {})
        report += f"Statistical Parity Difference: {disparities.get('statistical_parity_difference', 0):.1%}\n"
        report += f"Demographic Parity Ratio: {disparities.get('demographic_parity_ratio', 0):.3f}\n\n"
    
    violations = results_data.get('fairness_violations', [])
    if violations:
        report += "FAIRNESS VIOLATIONS\n"
        report += "==================\n"
        for violation in violations:
            report += f"- {violation.get('type', '')}: {violation.get('description', '')}\n"
    
    report += f"\n\nGenerated by Bias Buster Platform\n"
    return report

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Bias Buster Platform...")
    print("="*50)
    print(f"📊 GPT-5 nano integration: {'✅' if services['openai'] else '⚠️ Fallback'}")
    print(f"🔍 Bias detection: {'✅' if services['bias_detector'] else '⚠️ Fallback'}")
    print(f"📄 PDF report generation: {'✅' if services['pdf_generator'] else '⚠️ Fallback'}")
    print(f"🌐 Access at: http://localhost:5000")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
