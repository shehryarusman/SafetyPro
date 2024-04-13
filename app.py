import tkinter as tk
from tkinter import ttk
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class ToggleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hello Window")

        # Add a label
        self.label = ttk.Label(root, text="Hello!", font=("Helvetica", 16))
        self.label.pack(pady=20)

        # Add a start button
        self.start_button = ttk.Button(root, text="Start", command=self.start)
        self.start_button.pack(pady=10)

        # Add a stop button
        self.stop_button = ttk.Button(root, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

    def start(self):
        logging.info("Start button pressed")
        self.start_button['state'] = tk.DISABLED
        self.stop_button['state'] = tk.NORMAL

    def stop(self):
        logging.info("Stop button pressed")
        self.start_button['state'] = tk.NORMAL
        self.stop_button['state'] = tk.DISABLED

if __name__ == "__main__":
    root = tk.Tk()
    app = ToggleApp(root)
    root.mainloop()
