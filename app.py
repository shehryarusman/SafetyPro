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
        camera = dxcam.create()
        camera.start()
        while self.active:
            self.update_func()
            # cv2.imshow("Frame", frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # tic = time.perf_counter()
            # results = pytesseract.image_to_data(cv2.cvtColor(camera.grab(), cv2.COLOR_RGB2BGR), output_type=pytesseract.Output.DICT)
            # toc = time.perf_counter()
            # print("Time to convert with camera.grab  ", (toc - tic))
            # tic = time.perf_counter()
            # results = pytesseract.image_to_data(cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR), output_type=pytesseract.Output.DICT)
            # toc = time.perf_counter()
            # print("Time to convert with pyautogui ", (toc - tic))
            tic = time.perf_counter()
            results = pytesseract.image_to_data(cv2.cvtColor(camera.get_latest_frame(), cv2.COLOR_RGB2BGR), output_type=pytesseract.Output.DICT)
            toc = time.perf_counter()
            print("Time to convert with get_latest_frame ", (toc - tic))
            tic = time.perf_counter()
            n_boxes = len(results['text'])
            self.clear_func()

            for i in range(n_boxes):
                nlpResult = self.nlp(results['text'][i])
                if ((nlpResult[0]['label']=='toxic') and nlpResult[0]['score']>0.80):
                    self.hide_func(results['left'][i], results['top'][i] + 30, results['width'][i], results['height'][i])
            toc = time.perf_counter()
            print("Time to parse ", (toc - tic))
        camera.stop()


if __name__ == '__main__':
    root = tk.Tk()
    app = TextDetectionApp(root, None)
    root.mainloop()