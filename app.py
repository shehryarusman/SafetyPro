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

nthreads = 12

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

            tic = time.perf_counter()
            results = pytesseract.image_to_data(cv2.cvtColor(camera.get_latest_frame(), cv2.COLOR_RGB2GRAY), output_type=pytesseract.Output.DICT)
            toc = time.perf_counter()
            print("Time to convert with get_latest_frame ", (toc - tic))

            tic = time.perf_counter()

            n_boxes = len(results['text'])
            texts = results['text']
            texts = list(filter(None, texts))

            words_index = []
            t_threads = []
            index_results = [[] for i in range(nthreads)]
            split = n_boxes // nthreads
            start = 0

            for i in range(nthreads):
                #print(f'Thread: {i} \nresults:{results['text'][start:(start + split)]} \nStart: {start} \nSplit: {split}')
                if (i < (nthreads - 1)):
                    t_threads.append(threading.Thread(target=self.classify_thread_arr(results['text'][start:(start + split)], start, index_results[i])))
                else:
                    t_threads.append(threading.Thread(target=self.classify_thread_arr(results['text'][start:], start + (n_boxes % split), index_results[i])))
                start += split
                t_threads[i].start()

            for i in range(nthreads):
                t_threads[i].join()
                words_index += index_results[i]

            self.clear_func()
            for i in words_index:
                self.hide_func(results['left'][i], results['top'][i]+26, results['width'][i], results['height'][i])
            toc = time.perf_counter()
            print("Time to parse ", (toc - tic))
        camera.stop()

    def classify_thread_arr(self, text_arr, start, result):
        for i, text in enumerate(text_arr):
            nlpResult = self.nlp(text)
            if ((nlpResult[0]['label']=='toxic') and nlpResult[0]['score']>0.80):
                result.append(start + i)

if __name__ == '__main__':
    root = tk.Tk()
    app = TextDetectionApp(root, None)
    root.mainloop()