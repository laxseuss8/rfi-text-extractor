import pytesseract

# Hardcoded Tesseract path for Windows deployment
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def _extract_process(text):
    myconfig = r"--psm 11 --oem 3"
    text = pytesseract.image_to_string(text, config=myconfig).strip()
    return text

def extract_from_image(processed_x, processed_y):
    """Processes two processed images using the same steps."""
    extracted_x = _extract_process(processed_x)
    extracted_y = _extract_process(processed_y)
    return extracted_x, extracted_y
