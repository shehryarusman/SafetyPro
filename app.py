import logging
import shutil
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import pytesseract
import pyautogui
import dxcam
import threading
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import threading

threads = 8
logging.basicConfig(level=logging.INFO)

class TextDetectionApp:
    def __init__(self, hide_func, clear_func, update_func):
        self.hide_func = hide_func
        self.clear_func = clear_func
        self.update_func = update_func

        tesseract_path = shutil.which("tesseract")
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            pytesseract.pytesseract.tesseract_cmd ='C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

        model_name = "JungleLee/bert-toxic-comment-classification"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

        self.active = True
        self.detect_text()

    def stop_detection(self):
        self.active = False

    def detect_text(self):
        import time
        camera = dxcam.create()
        camera.start()
        while self.active:

            tick = time.perf_counter()
            results = pytesseract.image_to_data(cv2.cvtColor(camera.get_latest_frame(), cv2.COLOR_RGB2BGR), output_type=pytesseract.Output.DICT)
            tock = time.perf_counter()
            print(f'Screenshot Time: {tock - tick}')

            n_boxes = len(results['text'])

            #create sentences

            tick = time.perf_counter()
            bad_word_index = []
            t_threads = []
            index_results = [[] for i in range(threads)]
            split = n_boxes // threads
            start = 0;
            for i in range(threads):
                if (i < (threads - 1)):
                    t_threads.append(threading.Thread(target=self.classify_and_log_text_worker(results['text'][start:split], start, index_results[i], results)))
                else:
                    t_threads.append(threading.Thread(target=self.classify_and_log_text_worker(results['text'][start:], start + (n_boxes % split), index_results[i], results)))
                start += split
                t_threads[i].start()

            for i in range(threads):
                t_threads[i].join()
                bad_word_index += index_results[i]

            tock = time.perf_counter()
            print(f'Threading Time: {tock - tick}')
            current_line_text = ""
            last_y = 0
            self.update_func()
            self.clear_func()
        
                

            

    def classify_and_log_text(self, text, x, y, w, h):
        if self.is_inappropriate(text):
            #logging.info(f"Inappropriate text detected: {text} at position ({x}, {y}, {w}, {h})")
            return True
        else:
            #logging.info("Appropriate text detected: " + text)
            return False

    def classify_and_log_text_worker(self, text_arr, start, result, results):
        i = 0
        print("thread started")
        for text in text_arr:
            if self.classify_and_log_text(text, 0, 0, 0, 0):
                current_line_text=results["text"][i]
                print(current_line_text)
                last_x, last_y, last_w, last_h = results['left'][i], results['top'][i], results['width'][i], results['height'][i]
                if  self.hide_func != None:
                    self.hide_func(last_x, last_y+20, last_w, last_h)
                result.append(start + i)
            i += 1

    def is_inappropriate(self, text):
        if text.strip():
            results = self.nlp(text)
            if (results[0]['label']=='toxic') and results[0]['score']>0.85:
                return True
        return False

if __name__ == '__main__':
    root = tk.Tk()
    app = TextDetectionApp(root, None)
    root.mainloop()