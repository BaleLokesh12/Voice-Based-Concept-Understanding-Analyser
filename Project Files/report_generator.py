import os
import re
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def generate_pdf_waveform(analysis_results):
    """
    Generates a high-quality static waveform image and returns its temporary file path.
    """
    wf = analysis_results["waveform"]
    wf_t = analysis_results["waveform_times"]
    rms = analysis_results["rms_envelope"]
    rms_t = analysis_results["rms_times"]
    pauses = analysis_results["pauses"]
    
    # Create plot
    plt.figure(figsize=(7.5, 2.2))
    
    # Soft neutral background
    ax = plt.gca()
    ax.set_facecolor('#F8FAFC')
    
    # Plot waveform
    plt.plot(wf_t, wf, color='#60A5FA', alpha=0.6, label='Audio Waveform', linewidth=0.5)
    # Plot RMS Energy
    plt.plot(rms_t, rms, color='#EC4899', alpha=0.85, label='RMS Energy (Volume)', linewidth=1.5)
    
    # Highlight pauses
    for idx, (start, end, dur) in enumerate(pauses):
        plt.axvspan(start, end, color='#FCA5A5', alpha=0.3)
        
    plt.title("Audio Signal Profile (Waveform & Speaking Energy Envelope)", fontsize=9, color='#1E293B', fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=7, color='#475569')
    plt.ylabel("Amplitude", fontsize=7, color='#475569')
    plt.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
    plt.legend(fontsize=6.5, loc='upper right', frameon=True, facecolor='white', edgecolor='#E2E8F0')
    
    # Adjust boundaries
    plt.xlim(0, max(1.0, wf_t[-1] if len(wf_t) > 0 else 1.0))
    plt.tight_layout()
    
    # Save to a temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"vbcua_pdf_waveform_{int(datetime.now().timestamp())}.png")
    plt.savefig(temp_path, dpi=250, bbox_inches='tight')
    plt.close()
    return temp_path

def build_pdf_report(eval_results, analysis_results, student_name="Student Evaluator", filepath=None):
    """
    Compiles all analysis and semantic grading results into a beautiful ReportLab PDF report.
    Returns the absolute path to the generated PDF.
    """
    if filepath is None:
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, f"VBCUA_Report_{int(datetime.now().timestamp())}.pdf")
        
    # Setup document template with 0.5-inch margins for layout control
    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Color Palette
    primary_color = colors.HexColor('#1E3A8A')  # Dark Slate Blue
    secondary_color = colors.HexColor('#0D9488') # Teal
    dark_gray = colors.HexColor('#1E293B')      # Slate 800
    light_gray = colors.HexColor('#F8FAFC')     # Slate 50
    border_color = colors.HexColor('#E2E8F0')   # Slate 200
    
    # Set up custom styles
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20,
        textColor=primary_color, spaceAfter=4, alignment=TA_LEFT
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10,
        textColor=colors.HexColor('#64748B'), spaceAfter=15, alignment=TA_LEFT
    )
    section_heading = ParagraphStyle(
        'SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12,
        textColor=primary_color, spaceBefore=10, spaceAfter=6, borderPadding=(0,0,2,0)
    )
    body_style = ParagraphStyle(
        'ReportBody', parent=styles['BodyText'], fontName='Helvetica', fontSize=9,
        textColor=dark_gray, leading=13
    )
    bold_body_style = ParagraphStyle(
        'ReportBodyBold', parent=body_style, fontName='Helvetica-Bold'
    )
    transcription_style = ParagraphStyle(
        'TranscriptionText', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9,
        textColor=colors.HexColor('#334155'), leading=14
    )
    metric_label_style = ParagraphStyle(
        'MetricLabel', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9,
        textColor=colors.HexColor('#475569'), alignment=TA_LEFT
    )
    metric_value_style = ParagraphStyle(
        'MetricVal', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9,
        textColor=primary_color, alignment=TA_RIGHT
    )
    badge_style = ParagraphStyle(
        'BadgeText', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12,
        textColor=colors.white, alignment=TA_CENTER
    )
    
    # Header Section
    story.append(Paragraph("Voice-Based Concept Understanding Analyser", title_style))
    date_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    story.append(Paragraph(f"AI-Powered Educational Oral Assessment Report • Generated on {date_str}", subtitle_style))
    
    # Metadata Table
    metadata_data = [
        [
            Paragraph("<b>Participant:</b>", body_style), Paragraph(student_name, body_style),
            Paragraph("<b>Concept Evaluated:</b>", body_style), Paragraph(eval_results["concept_name"], bold_body_style)
        ],
        [
            Paragraph("<b>Audio Duration:</b>", body_style), Paragraph(f"{analysis_results['duration']:.2f} seconds", body_style),
            Paragraph("<b>Evaluation Method:</b>", body_style), Paragraph("Sentence-BERT Similarity", body_style)
        ]
    ]
    meta_table = Table(metadata_data, colWidths=[80, 180, 110, 170])
    meta_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,-1), (-1,-1), 1.5, primary_color),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # Core Scores Dashboard Grid
    overall_score = eval_results["overall_score"]
    level = eval_results["understanding_level"]
    
    # Setup level badge color
    if "Strong" in level:
        badge_bg = colors.HexColor('#059669') # Emerald 600
    elif "Moderate" in level:
        badge_bg = colors.HexColor('#D97706') # Amber 600
    elif "Basic" in level:
        badge_bg = colors.HexColor('#2563EB') # Blue 600
    else:
        badge_bg = colors.HexColor('#DC2626') # Red 600
        
    dashboard_data = [
        [
            Paragraph("<b>OVERALL VBCUA PERFORMANCE SCORE</b>", ParagraphStyle('OVS', parent=body_style, fontSize=8, textColor=colors.HexColor('#475569'), alignment=TA_CENTER)),
            Paragraph("<b>CONCEPT COMPREHENSION LEVEL</b>", ParagraphStyle('CCL', parent=body_style, fontSize=8, textColor=colors.HexColor('#475569'), alignment=TA_CENTER))
        ],
        [
            Paragraph(f"<font size=28 color='#1E3A8A'><b>{overall_score:.1f}/100</b></font>", ParagraphStyle('OVSScore', parent=body_style, alignment=TA_CENTER)),
            Table([[Paragraph(level.upper(), badge_style)]], colWidths=[200], rowHeights=[32], style=[
                ('BACKGROUND', (0,0), (-1,-1), badge_bg),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ])
        ]
    ]
    dash_table = Table(dashboard_data, colWidths=[270, 270])
    dash_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,-1), light_gray),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,0), 0.5, border_color),
    ]))
    story.append(dash_table)
    story.append(Spacer(1, 15))
    
    # Detailed Metrics Section
    story.append(Paragraph("Assessment Metrics Breakdown", section_heading))
    
    metrics_data = [
        [
            Paragraph("Concept Comprehension Score (60%)", metric_label_style),
            Paragraph(f"{eval_results['comprehension_score']:.1f}%", metric_value_style),
            Paragraph("Speech Fluency & Delivery Score (40%)", metric_label_style),
            Paragraph(f"{eval_results['fluency_score']:.1f}%", metric_value_style)
        ],
        [
            Paragraph("   • Semantic Similarity (S-BERT)", ParagraphStyle('Ind', parent=body_style, leftIndent=10)),
            Paragraph(f"{eval_results['semantic_similarity']:.1f}%", metric_value_style),
            Paragraph("   • Pace Tempo (WPM)", ParagraphStyle('Ind', parent=body_style, leftIndent=10)),
            Paragraph(f"{eval_results['fluency_details']['wpm']:.1f} WPM", metric_value_style)
        ],
        [
            Paragraph("   • Keyword Coverage Ratio", ParagraphStyle('Ind', parent=body_style, leftIndent=10)),
            Paragraph(f"{eval_results['keyword_coverage']:.1f}%", metric_value_style),
            Paragraph("   • Silence Pause Ratio", ParagraphStyle('Ind', parent=body_style, leftIndent=10)),
            Paragraph(f"{eval_results['fluency_details']['pause_ratio']*100:.1f}%", metric_value_style)
        ],
        [
            Paragraph("", body_style), Paragraph("", body_style),
            Paragraph("   • Filler Word Count", ParagraphStyle('Ind', parent=body_style, leftIndent=10)),
            Paragraph(str(eval_results['fluency_details']['filler_count']), metric_value_style)
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[200, 70, 200, 70])
    metrics_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#F1F5F9')),
        ('BACKGROUND', (2,0), (3,0), colors.HexColor('#F1F5F9')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 15))
    
    # Audio signal analysis diagram
    waveform_img_path = generate_pdf_waveform(analysis_results)
    if os.path.exists(waveform_img_path):
        story.append(Paragraph("Acoustic Signal Analysis", section_heading))
        story.append(Image(waveform_img_path, width=540, height=158))
        story.append(Spacer(1, 10))
        
    # Transcription text
    story.append(Paragraph("Speech Transcription Transcript", section_heading))
    trans_text = eval_results["transcription"] if eval_results["transcription"] else "[No speech transcribed]"
    
    # Format text to bold filler words in transcription
    fillers = list(eval_results['fluency_details']['filler_map'].keys())
    for f in fillers:
        if f == "you know":
            trans_text = re.sub(r'\b(you\s+know)\b', r'<b>\1</b>', trans_text, flags=re.IGNORECASE)
        else:
            trans_text = re.sub(r'\b(' + re.escape(f) + r')\b', r'<b>\1</b>', trans_text, flags=re.IGNORECASE)
            
    trans_para = Paragraph(f'"{trans_text}"', transcription_style)
    trans_table = Table([[trans_para]], colWidths=[540])
    trans_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(trans_table)
    story.append(Spacer(1, 15))
    
    # Reference Keyword Checklist
    story.append(Paragraph("Vocabulary & Keyword Analysis", section_heading))
    covered_kws = eval_results["covered_keywords"]
    missed_kws = eval_results["missed_keywords"]
    
    vocab_text = (
        f"<b>Covered Terminology ({len(covered_kws)}):</b> <font color='#059669'>{', '.join(covered_kws) if covered_kws else 'None'}</font><br/>"
        f"<b>Missed Key Terminology ({len(missed_kws)}):</b> <font color='#DC2626'>{', '.join(missed_kws) if missed_kws else 'None'}</font>"
    )
    vocab_para = Paragraph(vocab_text, body_style)
    vocab_table = Table([[vocab_para]], colWidths=[540])
    vocab_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(vocab_table)
    story.append(Spacer(1, 15))
    
    # Qualitative Feedback & Recommendations
    story.append(Paragraph("Evaluation Feedback & Action Items", section_heading))
    list_items = []
    for item in eval_results["feedback"]:
        list_items.append(ListItem(Paragraph(item, body_style), bulletColor=primary_color, value='circle'))
        
    feedback_list = ListFlowable(
        list_items,
        bulletType='bullet',
        start='circle',
        leftIndent=15,
        bulletOffsetY=0.5
    )
    story.append(feedback_list)
    
    # Build Document
    doc.build(story)
    
    # Clean up temp waveform image
    if os.path.exists(waveform_img_path):
        try:
            os.remove(waveform_img_path)
        except Exception:
            pass
            
    return filepath
