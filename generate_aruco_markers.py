"""
ArUco Marker Generator for Ball Balancing Table

This script generates the 4 ArUco markers needed for the static base plate calibration.
Creates both individual PNG files and a PDF with markers already at 4cm size.

Usage: python generate_aruco_markers.py
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os

# Try to import reportlab for PDF generation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ ReportLab not available - PDF generation disabled")
    print("   Install with: pip install reportlab")

def generate_markers():
    """Generate ArUco markers for base plate corners"""
    
    # Create ArUco dictionary (DICT_4X4_50 for reliability)
    # Handle both old and new OpenCV ArUco API
    try:
        # New OpenCV 4.7+ API
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    except AttributeError:
        try:
            # Older OpenCV API
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        except AttributeError:
            # Even older API
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    # Marker parameters
    marker_size_pixels = 400  # High resolution for crisp printing
    border_bits = 1          # Standard border size
    
    # Create output directory
    output_dir = "aruco_markers"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🎯 Generating ArUco Markers for Ball Balancing Table")
    print("=" * 50)
    print(f"📁 Output directory: {output_dir}")
    print(f"🖨️ Print size: 4cm x 4cm each")
    print(f"📐 Base plate size: 35cm x 35cm")
    print()
    
    # Generate markers 0, 1, 2, 3 for the four corners
    marker_positions = {
        0: "Top-Left Corner",
        1: "Top-Right Corner", 
        2: "Bottom-Right Corner",
        3: "Bottom-Left Corner"
    }
    
    for marker_id in range(4):
        # Generate marker image - handle different OpenCV versions
        try:
            # New OpenCV 4.7+ API
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels, borderBits=border_bits)
        except AttributeError:
            # Older OpenCV API
            marker_img = aruco.drawMarker(aruco_dict, marker_id, marker_size_pixels, borderBits=border_bits)
        
        # Add white border for easier cutting/placement
        border_size = 50
        bordered_img = cv2.copyMakeBorder(
            marker_img, border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=255
        )
        
        # Save marker
        filename = f"marker_{marker_id}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, bordered_img)
        
        position = marker_positions[marker_id]
        print(f"✅ Generated Marker {marker_id}: {filename} ({position})")
    
    # Generate PDF with correctly sized markers
    if PDF_AVAILABLE:
        generate_pdf_markers(aruco_dict, output_dir)
    
    print()
    print("🎯 SETUP INSTRUCTIONS:")
    print("=" * 30)
    if PDF_AVAILABLE:
        print("📄 RECOMMENDED: Print the PDF file for correct sizing!")
        print("   markers_instruction_sheet.pdf - Print and cut out markers")
        print()
    print("Alternative: Print individual markers at 4cm x 4cm size")
    print("2. Get a 35cm x 35cm wooden base plate")
    print("3. Place markers 2cm from each corner:")
    print("   • Marker 0: Top-Left corner")
    print("   • Marker 1: Top-Right corner") 
    print("   • Marker 2: Bottom-Right corner")
    print("   • Marker 3: Bottom-Left corner")
    print("4. Mount servo mechanism in CENTER of base")
    print("5. Your 25cm tilting plate goes above the servos")
    print("6. Run camera calibration: camera.calibrate_table_detection()")
    print()
    print("✅ All markers generated successfully!")

def generate_pdf_markers(aruco_dict, output_dir):
    """Generate PDF with markers at exactly 4cm size for easy printing"""
    if not PDF_AVAILABLE:
        return
    
    pdf_path = os.path.join(output_dir, "markers_instruction_sheet.pdf")
    
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
    story.append(Paragraph("Ball Balancing Table - ArUco Markers", title_style))
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
    story.append(Spacer(1, 1.5*cm))
    
    # Generate and add markers with proper spacing
    marker_size = 4*cm  # Exactly 4cm in ReportLab units
    gap_size = 2*cm     # 2cm gap between markers for easy cutting
    
    # Create temporary marker images
    marker_files = []
    for marker_id in range(4):
        # Generate marker image at high DPI for crisp PDF
        try:
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)
        except AttributeError:
            marker_img = aruco.drawMarker(aruco_dict, marker_id, 400)
        
        # Save temporary image
        temp_path = os.path.join(output_dir, f"temp_marker_{marker_id}.png")
        cv2.imwrite(temp_path, marker_img)
        marker_files.append(temp_path)
    
    # Create markers with proper spacing - 2x2 layout
    # First row: Marker 0 and 1
    row1_labels = [
        [Paragraph(f"<b>Marker 0</b><br/>(Top-Left)", styles['Normal']),
         "",  # Empty cell for spacing
         Paragraph(f"<b>Marker 1</b><br/>(Top-Right)", styles['Normal'])]
    ]
    
    row1_images = [
        [Image(marker_files[0], width=marker_size, height=marker_size),
         "",  # Empty cell for spacing
         Image(marker_files[1], width=marker_size, height=marker_size)]
    ]
    
    # Second row: Marker 3 and 2  
    row2_labels = [
        [Paragraph(f"<b>Marker 3</b><br/>(Bottom-Left)", styles['Normal']),
         "",  # Empty cell for spacing
         Paragraph(f"<b>Marker 2</b><br/>(Bottom-Right)", styles['Normal'])]
    ]
    
    row2_images = [
        [Image(marker_files[3], width=marker_size, height=marker_size),
         "",  # Empty cell for spacing
         Image(marker_files[2], width=marker_size, height=marker_size)]
    ]
    
    # Create tables with spacing
    col_widths = [marker_size, gap_size, marker_size]
    
    # First row labels and images
    label_table1 = Table(row1_labels, colWidths=col_widths)
    label_table1.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(label_table1)
    story.append(Spacer(1, 0.3*cm))
    
    img_table1 = Table(row1_images, colWidths=col_widths, rowHeights=[marker_size])
    img_table1.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (0, 0), 2, colors.black),  # Border for marker 0
        ('BOX', (2, 0), (2, 0), 2, colors.black),  # Border for marker 1
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(img_table1)
    story.append(Spacer(1, 2*cm))  # Gap between rows
    
    # Second row labels and images
    label_table2 = Table(row2_labels, colWidths=col_widths)
    label_table2.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(label_table2)
    story.append(Spacer(1, 0.3*cm))
    
    img_table2 = Table(row2_images, colWidths=col_widths, rowHeights=[marker_size])
    img_table2.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (0, 0), 2, colors.black),  # Border for marker 2
        ('BOX', (2, 0), (2, 0), 2, colors.black),  # Border for marker 3
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(img_table2)
    
    # Build PDF
    doc.build(story)
    
    # Clean up temporary files
    for temp_file in marker_files:
        try:
            os.remove(temp_file)
        except:
            pass
    
    print(f"📄 Generated PDF with correctly sized markers: markers_instruction_sheet.pdf")
    print(f"   🎯 Print at 100% scale - markers will be exactly 4cm x 4cm!")

if __name__ == "__main__":
    try:
        generate_markers()
    except ImportError:
        print("❌ OpenCV with ArUco support not installed")
        print("   Install with: pip install opencv-contrib-python")
    except Exception as e:
        print(f"❌ Error generating markers: {e}")
