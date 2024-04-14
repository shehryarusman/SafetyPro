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
import re

logging.basicConfig(level=logging.INFO)

class TextDetectionApp:
    def __init__(self):

        model_name = "JungleLee/bert-toxic-comment-classification"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        self.bad_phrase_cache = {}
        self.nthreads = 12
        self.pattern =  r"[a-zA-Z]+"

    def detect_text(self, results):
        n_boxes = len(results['text'])

        words_index = []
        t_threads = []
        index_results = [[] for _ in range(self.nthreads)]
        split = n_boxes // self.nthreads
        start = 0

        for i in range(self.nthreads):
            end = start + split if i < self.nthreads - 1 else n_boxes
            t_threads.append(threading.Thread(target=self.classify_thread_arr, args=(results['text'][start:end], start, index_results[i])))
            start = end

        for thread in t_threads:
            thread.start()

        for i in range(self.nthreads):
            t_threads[i].join()
            words_index += index_results[i]

        return words_index

    def classify_thread_arr(self, text_arr, start, result):
        for i, text in enumerate(text_arr):
            if text in self.bad_phrase_cache:
                if self.bad_phrase_cache[text]:
                    result.append(start + i)
                continue

            if not re.fullmatch(self.pattern, text):
                self.bad_phrase_cache[text] = False

            nlpResult = self.nlp(text)
            if ((nlpResult[0]['label']=='toxic') and nlpResult[0]['score']>0.80):
                result.append(start + i)
                self.bad_phrase_cache[text] = True
            else:
                self.bad_phrase_cache[text] = False

if __name__ == '__main__':
    root = tk.Tk()
    app = TextDetectionApp(root, None)
    root.mainloop()