from PIL import Image, ImageDraw, ImageFont

# Create an image with white background
width, height = 400, 100
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Load a font
try:
    font_path = "calibri.ttf"  # Ensure you have Arial font or change the path to an existing TTF font
    font_size_main = 36
    font_size_tagline = 18
    font_main = ImageFont.truetype(font_path, font_size_main)
    font_tagline = ImageFont.truetype(font_path, font_size_tagline)
except IOError:
    # Fallback to default font if the specified font is not available
    font_main = ImageFont.load_default()
    font_tagline = ImageFont.load_default()

# Draw the main text (center-aligned)
main_text = "synthetix.ai"
main_text_width, main_text_height = draw.textsize(main_text, font=font_main)
main_text_position = ((width - main_text_width) / 2, 10)
draw.text(main_text_position, main_text, font=font_main, fill='black')

# Draw the tagline (center-aligned)
tagline_text = "Unlocking Insights, Preserving Privacy"
tagline_text_width, tagline_text_height = draw.textsize(tagline_text, font=font_tagline)
tagline_text_position = ((width - tagline_text_width) / 2, main_text_position[1] + main_text_height + 10)
draw.text(tagline_text_position, tagline_text, font=font_tagline, fill='gray')

# Save the image
logo_path = "synthetix_logo.png"
image.save(logo_path)
print(f"Logo saved to {logo_path}")
