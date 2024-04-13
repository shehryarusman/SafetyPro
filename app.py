import logging
import shutil
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import pytesseract
import pyautogui
import threading
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Used for drawing over screen
import sys
from PyQt5.QtWidgets import QApplication
from screeninfo import get_monitors
from mainWindow import MainWindow



logging.basicConfig(level=logging.INFO)

class TextDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Real-Time Text Detection")

        self.active = False  
        self.thread = None   

        tesseract_path = shutil.which("tesseract")
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            raise EnvironmentError("Tesseract is not installed or not found in PATH.")

        model_name = "michellejieli/NSFW_text_classifier"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

        self.label = ttk.Label(master, text="Real-Time Text Detection", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.start_button = ttk.Button(master, text="Start", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.stop_button = ttk.Button(master, text="Stop", command=self.stop_detection, state='disabled')
        self.stop_button.pack(pady=5)

        # Handle to QT Window for drawing block over text
        self.textBlockQA = QApplication(sys.argv)
        self.textBlockWindow = MainWindow(get_monitors()[0].width, get_monitors()[0].height)
        self.textBlockWindow.show()
        self.textBlockQA.exec_()

    def start_detection(self):
        self.start_button['state'] = 'disabled'
        self.stop_button['state'] = 'normal'
        self.active = True
        self.detect_text() # Testing code
        #self.thread = threading.Thread(target=self.detect_text)
        #self.thread.start()

    def stop_detection(self):
        self.active = False
        #self.thread.join()
        self.start_button['state'] = 'normal'
        self.stop_button['state'] = 'disabled'

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
                if int(results['conf'][i]) > 60:
                    if abs(last_y - results['top'][i]) > 10:
                        if current_line_text:
                            self.classify_and_log_text(current_line_text, last_x, last_y, last_w, last_h)
                        current_line_text = results['text'][i]
                    else:
                        current_line_text += " " + results['text'][i]
                    
                    last_x, last_y, last_w, last_h = results['left'][i], results['top'][i], results['width'][i], results['height'][i]
            
            if current_line_text:
                toHide = self.classify_and_log_text(current_line_text, last_x, last_y, last_w, last_h)

            if toHide:
                self.textBlockWindow.addRect(last_x, last_y, last_w, last_h)
            

    def classify_and_log_text(self, text, x, y, w, h):
        logging.info(text)
        if self.is_inappropriate(text):
            logging.info(f"Inappropriate text detected: {text} at position ({x}, {y}, {w}, {h})")
            return True
        else:
            logging.info("Appropriate text detected: " + text)
            return False

    def is_inappropriate(self, text):
        if text.strip():
            results = self.nlp(text)
            if (results[0]['label']=='NSFW') and results[0]['score']>0.75:
                return True
        return False

if __name__ == '__main__':
    root = tk.Tk()
    app = TextDetectionApp(root)
    root.mainloop()