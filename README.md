# 🔍 Bias Buster - AI Fairness Audit Platform

> **Winner Solution for OpenAI x NxtWave Buildathon 2025**

An intelligent platform that detects AI bias in datasets using **GPT-5 nano** and generates professional audit reports.

![Bias Buster Platform](https://img.shields.io/badge/AI-GPT--5%20nano-blue) ![Status](https://img.shields.io/badge/Status-Production%20Ready-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Problem Statement

AI bias in hiring, lending, and decision-making systems causes real-world discrimination and costs organizations millions. Current bias detection tools are either too complex for business users or lack AI-powered insights.

## 💡 Our Solution

**Bias Buster** is an intelligent web platform that:
- 🤖 **AI-Powered Analysis**: Uses GPT-5 nano for expert bias insights
- 📊 **Comprehensive Detection**: Statistical parity, demographic parity, equality metrics
- 📄 **Professional Reports**: Downloadable PDF audit reports for compliance
- ⚡ **Real-time Processing**: Upload CSV → Get analysis in seconds
- 🎯 **Business-Ready**: Non-technical interface for HR, compliance, and legal teams

## 🛠️ Tech Stack

- **AI Model**: OpenAI GPT-5 nano (latest release)
- **Backend**: Python Flask + Custom Bias Detection Engine
- **Frontend**: Bootstrap 5 + Responsive Design
- **Data Processing**: Pandas + NumPy + Statistical Analysis
- **PDF Generation**: ReportLab with professional templates
- **Deployment**: Docker-ready with environment configs

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- OpenAI API key with GPT-5 nano access

### Installation

1. **Clone the repository**
git clone https://github.com/yourusername/bias-buster.git
cd bias-buster

text

2. **Set up virtual environment**
python -m venv venv

Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

text

3. **Install dependencies**
pip install -r requirements.txt

text

4. **Configure environment**
cp .env.example .env

Add your OpenAI API key to .env file
text

5. **Run the application**
cd apps/web-app
python app.py

text

6. **Open browser**
http://localhost:5000

text

## 📖 How to Use

1. **Upload Dataset**: CSV, Excel, or JSON files (max 16MB)
2. **Select Protected Attributes**: Choose demographic columns (gender, race, age, etc.)
3. **Set Target Variable**: Specify the outcome column to analyze for bias
4. **Get AI Analysis**: GPT-5 nano provides expert insights and recommendations
5. **Download Report**: Professional PDF audit report for compliance

## 📊 Sample Data

Try the platform with our sample dataset:
name,gender,age,experience,target,department
John,Male,25,2,1,Engineering
Sarah,Female,28,3,0,Engineering
Mike,Male,30,5,1,Engineering
Lisa,Female,26,2,0,Marketing

text

**Expected Results**: High bias detected (100% correlation between gender and outcomes)

## 🏗️ Architecture

bias-buster-platform/
├── apps/
│ ├── web-app/ # Flask web application
│ └── api-service/ # API configuration
├── packages/
│ ├── bias-detector/ # Core bias detection engine
│ ├── report-generator/ # PDF report generation
│ └── shared-utils/ # OpenAI GPT-5 nano integration
├── docs/ # Documentation
├── scripts/ # Deployment scripts
└── requirements.txt # Dependencies

text

## 🎥 Demo

[Add demo GIF or video link here]

**Key Features Demonstrated:**
- Real-time bias detection
- GPT-5 nano AI analysis
- Interactive results dashboard
- Professional PDF reporting

## 🏆 Key Achievements

- ✅ **100% Working GPT-5 nano Integration**: Latest OpenAI model with reasoning capabilities
- ✅ **Professional Bias Detection**: Statistical parity, demographic parity, fairness metrics
- ✅ **Enterprise-Ready Reports**: Comprehensive PDF audit reports with charts and recommendations
- ✅ **User-Friendly Interface**: No technical expertise required
- ✅ **Scalable Architecture**: Modular monorepo design for enterprise deployment

## 🎯 Business Impact

- **Risk Reduction**: Prevent costly discrimination lawsuits
- **Compliance**: Meet regulatory requirements (GDPR, Equal Credit Opportunity Act)
- **Reputation**: Ensure fair AI practices
- **Market Opportunity**: $15B AI ethics and fairness market

## 🧪 Testing

Run the test suite:
python test_bias_detector.py

text

## 📋 API Documentation

### Upload and Analyze
POST /upload
Content-Type: multipart/form-data

Parameters:

file: CSV/Excel/JSON file

protected_attributes[]: List of demographic columns

target_variable: Outcome column name

text

### Download Report
GET /download_report/<session_id>
Returns: PDF audit report

text

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

Built for **OpenAI x NxtWave Buildathon 2025**

- **Developer**: [Your Name]
- **Institution**: [Your College/Organization]
- **Contact**: [Your Email]

## 🙏 Acknowledgments

- **OpenAI** for GPT-5 nano access
- **NxtWave** for organizing the buildathon
- **Community** for feedback and testing

---

⭐ **Star this repository if you found it helpful!**

**Built with ❤️ for AI fairness and ethics**