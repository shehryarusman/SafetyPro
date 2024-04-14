import argparse
import os
import shutil
import tkinter as tk
from PIL import Image, ImageTk
import pytesseract

import OCR

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--crop', help="crop OCR area in pixels (two vals required): width height",
                        nargs=2, type=int, metavar='')

    parser.add_argument('-v', '--view_mode', help="view mode for OCR boxes display (default=1)",
                        default=1, type=int, metavar='')
    parser.add_argument('-sv', '--show_views', help="show the available view modes and descriptions",
                        action="store_true")

    parser.add_argument("-l", "--language",
                        help="code for tesseract language, use + to add multiple (ex: chi_sim+chi_tra)",
                        metavar='', default=None)
    parser.add_argument("-sl", "--show_langs", help="show list of tesseract (4.0+) supported langs",
                        action="store_true")
    parser.add_argument("-s", "--src", help="SRC video source for video capture",
                        default=0, type=int)

    args = parser.parse_args()

    tess_path = shutil.which("tesseract")
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path 
    else:
        pytesseract.pytesseract.tesseract_cmd ='C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

    OCR.tesseract_location(tess_path)
    OCR.ocr_stream(view_mode=args.view_mode, source=args.src, crop=args.crop, language=args.language)

if __name__ == '__main__':
    main()