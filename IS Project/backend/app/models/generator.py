from PIL import Image, ImageDraw
import os
import uuid

OUTPUT_DIR = "generated_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_followup(file1_obj, file2_obj) -> str:
    """
    Dummy predictive generator.
    Ignores uploaded MRI files, generates a simple placeholder image.
    """
    # Generate a blank image with text
    img = Image.new("RGB", (256, 256), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 120), "Predicted MRI", fill="black")

    # Save with random filename
    filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    img.save(output_path)

    return output_path
