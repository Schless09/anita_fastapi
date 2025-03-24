from PIL import Image, ImageDraw, ImageFont
import os

def generate_favicon():
    # Create a new image with a white background
    size = (32, 32)
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw a simple "A" in the center
    # Draw a filled circle as background
    circle_radius = 14
    circle_center = (16, 16)
    draw.ellipse(
        [
            circle_center[0] - circle_radius,
            circle_center[1] - circle_radius,
            circle_center[0] + circle_radius,
            circle_center[1] + circle_radius
        ],
        fill='#4A90E2'  # A nice blue color
    )
    
    # Draw "A" in white
    try:
        # Try to use Arial font if available
        font = ImageFont.truetype("Arial", 20)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Get text size to center it
    text = "A"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Draw the text centered
    draw.text(
        (
            (size[0] - text_width) // 2,
            (size[1] - text_height) // 2 - 2
        ),
        text,
        font=font,
        fill='white'
    )
    
    # Ensure static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Save as ICO file
    image.save('static/favicon.ico', format='ICO')

if __name__ == "__main__":
    generate_favicon() 