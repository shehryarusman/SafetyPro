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
import threading

threads = 4
logging.basicConfig(level=logging.INFO)

class TextDetectionApp:
    def __init__(self, hide_func, clear_func):
        self.hide_func = hide_func
        self.clear_func = clear_func
        # self.master = master
        # master.title("Real-Time Text Detection")

        self.active = False  
        self.thread = None   

        tesseract_path = shutil.which("tesseract")
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            pytesseract.pytesseract.tesseract_cmd ='C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

        model_name = "JungleLee/bert-toxic-comment-classification"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

        self.start_detection()


    def start_detection(self):
        self.active = True
        self.detect_text() # Testing code

    def stop_detection(self):
        self.active = False

    def detect_text(self):
        while self.active:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
            inverted_image = cv2.bitwise_not(binary_image)

            #results = pytesseract.image_to_data(cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR), output_type=pytesseract.Output.DICT)


            results = pytesseract.image_to_data(inverted_image, output_type=pytesseract.Output.DICT)


            n_boxes = len(results['text'])

            for i in range(n_boxes):
                print(results['text'][i])

            #create sentences

            bad_word_index = []
            t_threads = []
            index_results = [[] for i in range(threads)]
            split = n_boxes // threads
            start = 0;
            for i in range(threads):
                if (i < (threads - 1)):
                    t_threads.append(threading.Thread(target=self.classify_and_log_text_worker(results['text'][start:split], start, index_results[i])))
                else:
                    t_threads.append(threading.Thread(target=self.classify_and_log_text_worker(results['text'][start:], start + (n_boxes % split), index_results[i])))
                start += split
                t_threads[i].start()

            for i in range(threads):
                t_threads[i].join()
                print(index_results[i])
                bad_word_index += index_results[i]

            current_line_text = ""
            last_y = 0
            # self.clear_func()
            for i in bad_word_index:
                current_line_text=results["text"][i]
                last_x, last_y, last_w, last_h = results['left'][i], results['top'][i], results['width'][i], results['height'][i]
                if  self.hide_func != None:
                    print(current_line_text)
                    self.hide_func(last_x, last_y+20, last_w, last_h)
        
                    

            

    def classify_and_log_text(self, text, x, y, w, h):
        if self.is_inappropriate(text):
            #logging.info(f"Inappropriate text detected: {text} at position ({x}, {y}, {w}, {h})")
            return True
        else:
            #logging.info("Appropriate text detected: " + text)
            return False

    def classify_and_log_text_worker(self, text_arr, start, result):
        i = 0
        for text in text_arr:
            if self.classify_and_log_text(text, 0, 0, 0, 0):
                result.append(start + i)
            i += 1

    def is_inappropriate(self, text):
        if text.strip():
            results = self.nlp(text)
            if (results[0]['label']=='toxic') and results[0]['score']>0.85:
                print(text)
                return True
        return False

if __name__ == '__main__':
    root = tk.Tk()
    app = TextDetectionApp(root, None)
    root.mainloop()