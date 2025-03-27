from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

def create_pdf():
    c = canvas.Canvas("test_resume.pdf", pagesize=letter)
    width, height = letter
    
    # Add content
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Harrison Franke")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, "Software Engineer")
    c.drawString(50, height - 90, "harrison.franke@gmail.com")
    c.drawString(50, height - 110, "+19299446066")
    
    # Experience
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 140, "Experience:")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 160, "• Software Engineer at Tech Corp (2020-2023)")
    c.drawString(50, height - 180, "• Software Engineer at StartUp Inc (2018-2020)")
    
    # Skills
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 210, "Skills:")
    
    c.setFont("Helvetica", 12)
    skills = ["Python", "JavaScript", "React", "Node.js", "AWS", "Docker"]
    y = height - 230
    for skill in skills:
        c.drawString(50, y, f"• {skill}")
        y -= 20
    
    # Education
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y - 20, "Education:")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, y - 40, "• BS Computer Science, University of Technology (2018)")
    
    c.save()

if __name__ == "__main__":
    create_pdf() 