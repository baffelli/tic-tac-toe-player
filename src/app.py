import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.


import numpy as np

import board

import matplotlib.pyplot as plt

class App:

    def __init__(self, brd: board.Board) -> None:

        self.root = tkinter.Tk()
        self.root.wm_title("Embedding in Tk")

        f = plt.figure()
        self.ax = f.add_subplot()


        self.canvas = FigureCanvasTkAgg(f, master=self.root)  # A tk.DrawingArea.
        brd.show(self.ax)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


        self.board = brd

    def draw(self):
        self.board.show(self.ax)



