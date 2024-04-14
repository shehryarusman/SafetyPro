import tkinter as tk
import pyautogui
import threading
import logging
import numpy as np
import cv2
import pytesseract
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logging.basicConfig(level=logging.INFO)

class TextDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Text Detection")
        self.geometry("800x600")  # Adjust size as needed for your screen

        # Start and Stop buttons
        self.start_button = tk.Button(self, text="Start", command=self.start_detection)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self, text="Stop", command=self.stop_detection, state='disabled')
        self.stop_button.pack(pady=10)

        # Canvas for drawing rectangles
        self.canvas = tk.Canvas(self, bg='white', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize the NLP model
        model_name = "michellejieli/NSFW_text_classifier"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

        self.active = False
        self.thread = None

    def start_detection(self):
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.active = True
        self.thread = threading.Thread(target=self.detect_text)
        self.thread.start()

    def stop_detection(self):
        self.active = False
        self.stop_button.config(state='disabled')
        self.start_button.config(state='normal')
        self.quit()  # Close the application



    def detect_text(self):
        while self.active:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)
            n_boxes = len(results['text'])
            current_line_text = ""
            last_y = 0
            for i in range(n_boxes):
                if not self.active:  # Check if we should still be running
                    break
                if int(results['conf'][i]) > 80:
                    # Group text that are on the same or close Y coordinate
                    if abs(last_y - results['top'][i]) > 10:  # New line if Y difference is significant
                        if current_line_text:
                            self.classify_and_log_text(current_line_text, last_x//2, last_y//2, last_w//2, last_h//2)
                        current_line_text = results['text'][i]
                    else:
                        current_line_text += " " + results['text'][i]
                    
                    last_x, last_y, last_w, last_h = results['left'][i], results['top'][i], results['width'][i], results['height'][i]
            
            if current_line_text:
                self.classify_and_log_text(current_line_text, last_x, last_y, last_w, last_h)
            
    def classify_and_log_text(self, text, x, y, w, h):
        if self.is_inappropriate(text):
            self.draw_rectangle(x, y, w, h)
            logging.info(f"Inappropriate text detected: {text} at position ({x}, {y}, {w}, {h})")
        else:
            logging.info("Appropriate text detected: " + text)

    def draw_rectangle(self, x, y, w, h):
            # Create a separate window for the rectangle overlay
            logging.info(f"Drawing rectangle at ({x}, {y}, {w}, {h})")
            self.overlay = tk.Toplevel(self)
            self.overlay.geometry("{}x{}+{}+{}".format(w,h,x,y))
            self.overlay.overrideredirect(True)  # No window decorations
            self.overlay.attributes('-topmost', True)  # Always on top
            self.overlay.config(bg='red')  

    def is_inappropriate(self, text):
        if text.strip():
            results = self.nlp(text)
            if (results[0]['label'] == 'NSFW') and (results[0]['score'] > 0.92):
                return True
        return False

if __name__ == '__main__':
    app = TextDetectionApp()
    app.mainloop()
