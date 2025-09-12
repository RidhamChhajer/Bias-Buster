import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, 
                               TableStyle, PageBreak, Image, KeepTogether)
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

class PDFReportGenerator:
    """Generate comprehensive PDF bias audit reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50')
        )
        
        # Subtitle style
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.HexColor('#34495e'),
            alignment=TA_LEFT
        )
        
        # Risk level styles
        self.risk_styles = {
            'CRITICAL': ParagraphStyle(
                'CriticalRisk',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#dc3545'),
                backColor=colors.HexColor('#f8d7da'),
                borderColor=colors.HexColor('#dc3545'),
                borderWidth=1,
                borderPadding=10,
                alignment=TA_CENTER
            ),
            'HIGH': ParagraphStyle(
                'HighRisk',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#fd7e14'),
                backColor=colors.HexColor('#fff3cd'),
                borderColor=colors.HexColor('#fd7e14'),
                borderWidth=1,
                borderPadding=10,
                alignment=TA_CENTER
            ),
            'MEDIUM': ParagraphStyle(
                'MediumRisk',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#ffc107'),
                backColor=colors.HexColor('#fff9e6'),
                borderColor=colors.HexColor('#ffc107'),
                borderWidth=1,
                borderPadding=10,
                alignment=TA_CENTER
            ),
            'LOW': ParagraphStyle(
                'LowRisk',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#17a2b8'),
                backColor=colors.HexColor('#e6f7ff'),
                borderColor=colors.HexColor('#17a2b8'),
                borderWidth=1,
                borderPadding=10,
                alignment=TA_CENTER
            ),
            'MINIMAL': ParagraphStyle(
                'MinimalRisk',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#28a745'),
                backColor=colors.HexColor('#d4edda'),
                borderColor=colors.HexColor('#28a745'),
                borderWidth=1,
                borderPadding=10,
                alignment=TA_CENTER
            )
        }
    
    def generate_bias_report(self, analysis_data: Dict[str, Any], output_path: Path) -> bool:
        """Generate comprehensive bias audit PDF report"""
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build report content
            story = []
            
            # Title page
            story.extend(self._create_title_page(analysis_data))
            
            # Executive summary
            story.extend(self._create_executive_summary(analysis_data))
            
            # Dataset overview
            story.extend(self._create_dataset_overview(analysis_data))
            
            # Bias analysis results
            story.extend(self._create_bias_analysis(analysis_data))
            
            # AI insights
            story.extend(self._create_ai_insights(analysis_data))
            
            # Recommendations
            story.extend(self._create_recommendations(analysis_data))
            
            # Technical details
            story.extend(self._create_technical_details(analysis_data))
            
            # Build PDF
            doc.build(story)
            
            return True
            
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")
            return False
    
    def _create_title_page(self, data: Dict) -> List:
        """Create title page"""
        story = []
        
        # Main title
        story.append(Paragraph("AI BIAS AUDIT REPORT", self.title_style))
        story.append(Spacer(1, 20))
        
        # Dataset info
        dataset_info = data.get('dataset_info', {})
        filename = dataset_info.get('filename', 'Unknown Dataset')
        
        story.append(Paragraph(f"<b>Dataset:</b> {filename}", self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Report metadata
        timestamp = data.get('timestamp', datetime.now().isoformat())
        formatted_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%B %d, %Y at %I:%M %p')
        
        story.append(Paragraph(f"<b>Generated:</b> {formatted_date}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Analysis Engine:</b> Bias Buster Platform", self.styles['Normal']))
        
        ai_analysis = data.get('ai_analysis', {})
        model_used = ai_analysis.get('model_used', 'GPT-5-nano')
        story.append(Paragraph(f"<b>AI Model:</b> {model_used}", self.styles['Normal']))
        
        story.append(Spacer(1, 40))
        
        # Risk level prominently displayed
        risk_assessment = data.get('ai_analysis', {}).get('risk_level', 'UNKNOWN')
        bias_score = data.get('ai_analysis', {}).get('bias_score', 0.0)
        
        risk_style = self.risk_styles.get(risk_assessment, self.styles['Normal'])
        story.append(Paragraph(
            f"<b>OVERALL RISK LEVEL: {risk_assessment}</b><br/>Bias Score: {bias_score:.3f}",
            risk_style
        ))
        
        story.append(PageBreak())
        return story
    
    def _create_executive_summary(self, data: Dict) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("EXECUTIVE SUMMARY", self.subtitle_style))
        
        # Key findings
        ai_analysis = data.get('ai_analysis', {})
        analysis_text = ai_analysis.get('ai_analysis', 'No AI analysis available')
        
        # Extract summary from AI analysis (first few sentences)
        summary_sentences = analysis_text.split('.')[:3]
        summary_text = '. '.join(summary_sentences) + '.'
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key metrics table
        key_metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Overall Bias Score', f"{ai_analysis.get('bias_score', 0):.3f}", self._get_status(ai_analysis.get('bias_score', 0))],
            ['Risk Level', ai_analysis.get('risk_level', 'Unknown'), self._get_risk_color(ai_analysis.get('risk_level', 'Unknown'))],
            ['Protected Attributes', len(data.get('dataset_info', {}).get('protected_attributes', [])), '✓'],
            ['Total Records', f"{data.get('dataset_info', {}).get('total_records', 0):,}", '✓']
        ]
        
        key_metrics_table = Table(key_metrics_data, colWidths=[2*inch, 1.5*inch, 1*inch])
        key_metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(key_metrics_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_dataset_overview(self, data: Dict) -> List:
        """Create dataset overview section"""
        story = []
        
        story.append(Paragraph("DATASET OVERVIEW", self.subtitle_style))
        
        dataset_info = data.get('dataset_info', {})
        
        # Dataset statistics
        overview_data = [
            ['Attribute', 'Value'],
            ['Filename', dataset_info.get('filename', 'Unknown')],
            ['Total Records', f"{dataset_info.get('total_records', 0):,}"],
            ['Total Columns', len(dataset_info.get('columns', []))],
            ['Target Variable', dataset_info.get('target_variable', 'Unknown')],
            ['Protected Attributes', ', '.join(dataset_info.get('protected_attributes', []))]
        ]
        
        overview_table = Table(overview_data, colWidths=[2*inch, 3*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_bias_analysis(self, data: Dict) -> List:
        """Create detailed bias analysis section"""
        story = []
        
        story.append(Paragraph("BIAS ANALYSIS RESULTS", self.subtitle_style))
        
        bias_metrics = data.get('bias_metrics', {})
        
        for attr, metrics in bias_metrics.items():
            story.append(Paragraph(f"<b>{attr.upper()} ANALYSIS</b>", self.styles['Heading3']))
            
            # Group statistics
            groups = metrics.get('groups', {})
            group_data = [['Group', 'Sample Size', 'Positive Rate', 'Representation']]
            
            for group, stats in groups.items():
                group_data.append([
                    group,
                    f"{stats.get('group_size', 0):,}",
                    f"{stats.get('positive_rate', 0):.1%}",
                    f"{stats.get('representation', 0):.1%}"
                ])
            
            group_table = Table(group_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            group_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(group_table)
            story.append(Spacer(1, 15))
            
            # Disparity metrics
            disparities = metrics.get('disparities', {})
            story.append(Paragraph(f"<b>Disparity Metrics:</b>", self.styles['Normal']))
            story.append(Paragraph(f"Statistical Parity Difference: {disparities.get('statistical_parity_difference', 0):.1%}", self.styles['Normal']))
            story.append(Paragraph(f"Demographic Parity Ratio: {disparities.get('demographic_parity_ratio', 0):.3f}", self.styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Violations summary
        violations = data.get('fairness_violations', [])
        if violations:
            story.append(Paragraph("FAIRNESS VIOLATIONS", self.styles['Heading3']))
            
            violation_data = [['Type', 'Attribute', 'Severity', 'Description']]
            for violation in violations:
                violation_data.append([
                    violation.get('type', '').replace('_', ' ').title(),
                    violation.get('attribute', ''),
                    violation.get('severity', '').upper(),
                    violation.get('description', '')
                ])
            
            violation_table = Table(violation_data, colWidths=[1.5*inch, 1*inch, 0.8*inch, 2.5*inch])
            violation_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fadbd8')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            story.append(violation_table)
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_ai_insights(self, data: Dict) -> List:
        """Create AI insights section"""
        story = []
        
        story.append(Paragraph("AI EXPERT ANALYSIS", self.subtitle_style))
        
        ai_analysis = data.get('ai_analysis', {})
        analysis_text = ai_analysis.get('ai_analysis', 'No AI analysis available')
        
        # Split analysis into paragraphs for better formatting
        paragraphs = analysis_text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), self.styles['Normal']))
                story.append(Spacer(1, 12))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_recommendations(self, data: Dict) -> List:
        """Create recommendations section"""
        story = []
        
        story.append(Paragraph("RECOMMENDATIONS", self.subtitle_style))
        
        ai_analysis = data.get('ai_analysis', {})
        recommendations = ai_analysis.get('recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"<b>{i}.</b> {rec}", self.styles['Normal']))
                story.append(Spacer(1, 10))
        else:
            story.append(Paragraph("No specific recommendations available.", self.styles['Normal']))
        
        # General recommendations based on risk level
        risk_level = ai_analysis.get('risk_level', 'UNKNOWN')
        general_rec = self._get_general_recommendations(risk_level)
        
        if general_rec:
            story.append(Paragraph("<b>General Recommendations:</b>", self.styles['Heading3']))
            story.append(Paragraph(general_rec, self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_technical_details(self, data: Dict) -> List:
        """Create technical details section"""
        story = []
        
        story.append(Paragraph("TECHNICAL DETAILS", self.subtitle_style))
        
        ai_analysis = data.get('ai_analysis', {})
        
        tech_data = [
            ['Detail', 'Value'],
            ['Analysis Timestamp', data.get('timestamp', 'Unknown')],
            ['AI Model Used', ai_analysis.get('model_used', 'Unknown')],
            ['Parameters Used', str(ai_analysis.get('parameters_used', {}))],
            ['Token Usage', str(ai_analysis.get('token_usage', 'N/A'))],
            ['Session ID', data.get('session_id', 'Unknown')]
        ]
        
        tech_table = Table(tech_data, colWidths=[2*inch, 3*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#95a5a6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        
        story.append(tech_table)
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Paragraph(
            "<i>This report was generated by the Bias Buster Platform using advanced AI bias detection algorithms. "
            "For questions or support, please contact your system administrator.</i>",
            self.styles['Normal']
        ))
        
        return story
    
    def _get_status(self, bias_score: float) -> str:
        """Get status indicator based on bias score"""
        if bias_score >= 0.7:
            return "🔴 Critical"
        elif bias_score >= 0.5:
            return "🟠 High Risk"
        elif bias_score >= 0.3:
            return "🟡 Medium Risk"
        elif bias_score >= 0.1:
            return "🔵 Low Risk"
        else:
            return "🟢 Minimal Risk"
    
    def _get_risk_color(self, risk_level: str) -> str:
        """Get colored risk level indicator"""
        colors_map = {
            'CRITICAL': '🔴 Critical',
            'HIGH': '🟠 High', 
            'MEDIUM': '🟡 Medium',
            'LOW': '🔵 Low',
            'MINIMAL': '🟢 Minimal'
        }
        return colors_map.get(risk_level, '⚪ Unknown')
    
    def _get_general_recommendations(self, risk_level: str) -> str:
        """Get general recommendations based on risk level"""
        recommendations = {
            'CRITICAL': "Immediate intervention required. Consider data rebalancing, algorithmic debiasing, and comprehensive fairness auditing before any deployment.",
            'HIGH': "Implement bias mitigation strategies including data preprocessing, fairness constraints in model training, and ongoing monitoring.",
            'MEDIUM': "Apply standard bias reduction techniques and establish regular monitoring protocols. Consider fairness-aware machine learning approaches.",
            'LOW': "Maintain current practices with regular bias monitoring. Document findings for compliance purposes.",
            'MINIMAL': "Continue with existing monitoring practices. Use this analysis as a baseline for future comparisons."
        }
        return recommendations.get(risk_level, "")
