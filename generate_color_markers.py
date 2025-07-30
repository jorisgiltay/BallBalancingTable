"""
Blue Marker Generator for Ball Balancing Table

This script generates 4 BLUE markers for the static base plate calibration.
Much simpler and more reliable than ArUco markers!

All markers are BLUE - position determines identity:
- Blue (0): Top-Left corner (Red position)
- Blue (1): Top-Right corner (Green position)  
- Blue (2): Bottom-Right corner (Blue position)
- Blue (3): Bottom-Left corner (Magenta position)

Usage: python generate_color_markers.py
"""

import cv2
import numpy as np
import os

# Try to import reportlab for PDF generation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors as pdf_colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.graphics.shapes import Drawing, Rect
    from reportlab.graphics import renderPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ ReportLab not available - PDF generation disabled")
    print("   Install with: pip install reportlab")

def generate_blue_markers():
    """Generate 4 BLUE markers for base plate corners"""
    
    # Marker parameters
    marker_size = 400  # 400x400 pixels for crisp printing
    
    # All markers are blue - position determines identity
    marker_colors = {
        0: {"name": "Blue", "bgr": (255, 0, 0), "position": "Top-Left"},
        1: {"name": "Blue", "bgr": (255, 0, 0), "position": "Top-Right"},
        2: {"name": "Blue", "bgr": (255, 0, 0), "position": "Bottom-Right"},
        3: {"name": "Blue", "bgr": (255, 0, 0), "position": "Bottom-Left"}
    }
    
    # Create output directory
    output_dir = "blue_markers"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔵 Generating Blue Markers for Ball Balancing Table")
    print("=" * 50)
    print(f"📁 Output directory: {output_dir}")
    print(f"🖨️ Print size: 4cm x 4cm each")
    print(f"📐 Base plate size: 35cm x 35cm")
    print("🔵 ALL MARKERS ARE BLUE - Position determines identity!")
    print()
    
    for marker_id, info in marker_colors.items():
        # Create solid color square
        marker_img = np.full((marker_size, marker_size, 3), info["bgr"], dtype=np.uint8)
        
        # Add white border for easier cutting/placement
        border_size = 20
        bordered_img = cv2.copyMakeBorder(
            marker_img, border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        # Add text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Calculate text position (center)
        text = str(marker_id)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (bordered_img.shape[1] - text_width) // 2
        text_y = (bordered_img.shape[0] + text_height) // 2
        
        # Add white text on blue background for visibility
        text_color = (255, 255, 255)
        cv2.putText(bordered_img, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Save marker
        filename = f"marker_{marker_id}_blue.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, bordered_img)
        
        print(f"✅ Generated Marker {marker_id}: {filename} (Blue - {info['position']})")
    
    # Generate PDF with correctly sized markers
    if PDF_AVAILABLE:
        generate_pdf_blue_markers(marker_colors, output_dir)
    
    print()
    print("🎯 SETUP INSTRUCTIONS:")
    print("=" * 30)
    if PDF_AVAILABLE:
        print("📄 RECOMMENDED: Print the PDF file for correct sizing!")
        print("   blue_markers_instruction_sheet.pdf")
        print()
    print("Alternative: Print individual markers at 4cm x 4cm size")
    print("2. Get a 35cm x 35cm wooden base plate")
    print("3. Place markers 2cm from each corner:")
    print("   • Blue (0): Top-Left corner")
    print("   • Blue (1): Top-Right corner") 
    print("   • Blue (2): Bottom-Right corner")
    print("   • Blue (3): Bottom-Left corner")
    print("4. Mount servo mechanism in CENTER of base")
    print("5. Your 25cm tilting plate goes above the servos")
    print("6. Run camera calibration: camera.calibrate_table_detection()")
    print()
    print("✅ All BLUE markers generated successfully!")
    print("🔵 Position-based detection is much more reliable!")

def generate_pdf_blue_markers(marker_colors, output_dir):
    """Generate PDF with BLUE markers at exactly 4cm size"""
    if not PDF_AVAILABLE:
        return
    
    pdf_path = os.path.join(output_dir, "blue_markers_instruction_sheet.pdf")
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Ball Balancing Table - Blue Markers (Position-Based)", title_style))
    story.append(Spacer(1, 1*cm))
    
    # Brief instruction
    instruction_style = ParagraphStyle(
        'Instructions',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("<b>Print at 100% scale - Each marker will be exactly 4cm x 4cm</b>", instruction_style))
    story.append(Paragraph("<b>ALL MARKERS ARE BLUE - Position determines identity!</b>", instruction_style))
    story.append(Spacer(1, 1.5*cm))
    
    # Generate markers with proper spacing
    marker_size = 4*cm  # Exactly 4cm in ReportLab units
    gap_size = 2*cm     # 2cm gap between markers for easy cutting
    
    # Create colored rectangles for PDF - all blue!
    from reportlab.graphics.shapes import Drawing, Rect
    from reportlab.graphics import renderPDF
    
    # All markers are blue
    pdf_color = pdf_colors.blue
    
    # Create drawings for each marker
    marker_drawings = []
    for marker_id in range(4):
        drawing = Drawing(marker_size, marker_size)
        rect = Rect(0, 0, marker_size, marker_size)
        rect.fillColor = pdf_color
        rect.strokeColor = pdf_colors.black
        rect.strokeWidth = 2
        drawing.add(rect)
        marker_drawings.append(drawing)
    
    # Create markers with proper spacing - 2x2 layout
    col_widths = [marker_size, gap_size, marker_size]
    
    # First row: Marker 0 and 1
    row1_labels = [
        [Paragraph(f"<b>Blue (0)</b><br/>(Top-Left)", styles['Normal']),
         "",  # Empty cell for spacing
         Paragraph(f"<b>Blue (1)</b><br/>(Top-Right)", styles['Normal'])]
    ]
    
    row1_markers = [
        [marker_drawings[0], "", marker_drawings[1]]
    ]
    
    # Second row: Marker 3 and 2  
    row2_labels = [
        [Paragraph(f"<b>Blue (3)</b><br/>(Bottom-Left)", styles['Normal']),
         "",  # Empty cell for spacing
         Paragraph(f"<b>Blue (2)</b><br/>(Bottom-Right)", styles['Normal'])]
    ]
    
    row2_markers = [
        [marker_drawings[3], "", marker_drawings[2]]
    ]
    
    # Create tables with spacing
    # First row labels and markers
    label_table1 = Table(row1_labels, colWidths=col_widths)
    label_table1.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(label_table1)
    story.append(Spacer(1, 0.3*cm))
    
    marker_table1 = Table(row1_markers, colWidths=col_widths, rowHeights=[marker_size])
    marker_table1.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (0, 0), 2, pdf_colors.black),  # Border for marker 0
        ('BOX', (2, 0), (2, 0), 2, pdf_colors.black),  # Border for marker 1
    ]))
    story.append(marker_table1)
    story.append(Spacer(1, 2*cm))  # Gap between rows
    
    # Second row labels and markers
    label_table2 = Table(row2_labels, colWidths=col_widths)
    label_table2.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(label_table2)
    story.append(Spacer(1, 0.3*cm))
    
    marker_table2 = Table(row2_markers, colWidths=col_widths, rowHeights=[marker_size])
    marker_table2.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (0, 0), 2, pdf_colors.black),  # Border for marker 2
        ('BOX', (2, 0), (2, 0), 2, pdf_colors.black),  # Border for marker 3
    ]))
    story.append(marker_table2)
    
    # Build PDF
    doc.build(story)
    
    print(f"📄 Generated PDF with correctly sized BLUE markers: blue_markers_instruction_sheet.pdf")
    print(f"   🎯 Print at 100% scale - markers will be exactly 4cm x 4cm!")
    print(f"   🔵 All markers are blue - position determines identity!")

if __name__ == "__main__":
    try:
        generate_blue_markers()
    except Exception as e:
        print(f"❌ Error generating blue markers: {e}")
