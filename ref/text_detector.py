import pyautogui
import pytesseract
from PIL import Image

# OPTIONAL: Only needed if you're on Windows and Tesseract isn't in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Take a screenshot
screenshot = pyautogui.screenshot()

# Convert screenshot to something Tesseract can read
screenshot_rgb = screenshot.convert('RGB')

# Get bounding boxes for all detected words
data = pytesseract.image_to_data(screenshot_rgb, output_type=pytesseract.Output.DICT)

target_word = "Battle"  # Replace with the word you want to detect

for i in range(len(data['text'])):
    word = data['text'][i]
    if word.strip().lower() == target_word.lower():
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        print(f"Found '{target_word}' at: X: {x}, Y: {y}, Width: {w}, Height: {h}")
