import logging
import shutil
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import pytesseract
import pygetwindow as gw
import pyautogui
import threading

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

        self.label = ttk.Label(master, text="Real-Time Text Detection", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.start_button = ttk.Button(master, text="Start", command=self.start_detection)
        self.start_button.pack(pady=5)

        self.stop_button = ttk.Button(master, text="Stop", command=self.stop_detection, state='disabled')
        self.stop_button.pack(pady=5)

    def start_detection(self):
        self.start_button['state'] = 'disabled'
        self.stop_button['state'] = 'normal'
        self.active = True
        self.thread = threading.Thread(target=self.detect_text)
        self.thread.start()

    def stop_detection(self):
        self.active = False
        self.thread.join()
        self.start_button['state'] = 'normal'
        self.stop_button['state'] = 'disabled'

    def detect_text(self):
        while self.active:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            text = pytesseract.image_to_string(frame)
            logging.info(f"Detected text: {text}")


if __name__ == '__main__':
    root = tk.Tk()
    app = TextDetectionApp(root)
    root.mainloop()
