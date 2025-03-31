from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_test_resume():
    c = canvas.Canvas("test_resume.pdf", pagesize=letter)
    
    # Add content
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "John Doe")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, 730, "Software Engineer")
    c.drawString(50, 710, "Email: john.doe@example.com")
    c.drawString(50, 690, "Phone: +1234567890")
    c.drawString(50, 670, "LinkedIn: linkedin.com/in/johndoe")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 640, "Experience")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, 620, "Senior Software Engineer - Tech Corp (2020-Present)")
    c.drawString(70, 600, "• Led development of microservices architecture")
    c.drawString(70, 580, "• Implemented CI/CD pipeline")
    
    c.drawString(50, 550, "Software Engineer - Startup Inc (2018-2020)")
    c.drawString(70, 530, "• Developed full-stack web applications")
    c.drawString(70, 510, "• Built RESTful APIs")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 480, "Education")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, 460, "BS Computer Science - University of Technology (2018)")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 430, "Skills")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, 410, "• Python, JavaScript, React, Node.js")
    c.drawString(50, 390, "• AWS, Docker, Kubernetes")
    c.drawString(50, 370, "• SQL, NoSQL, GraphQL")
    c.drawString(50, 350, "• Git, CI/CD, Agile")
    
    c.save()

if __name__ == "__main__":
    create_test_resume() 