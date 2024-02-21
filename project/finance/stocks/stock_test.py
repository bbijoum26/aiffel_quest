# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from pykrx import stock
import pandas_datareader.data as pdr
import yfinance as yf

# GUI
import tkinter as tk
import tkinter.ttk
import tkinter.messagebox as msgbox
from tkinter import ttk
from tkinter import *
from tksheet import Sheet
from tkinter import filedialog as fd

# tkinter
root = tk.Tk()
root.title('stock price prediction')
notebook = tkinter.ttk.Notebook(root, width=1280, height=600)
notebook.pack()

# 파일 표시
label = Label(root)
label.pack()

file_frame = LabelFrame(root, text='Open File Dialog')
file_frame.pack(fill='x', padx=5, pady=5)

browse = Label(file_frame, text='No File Selected')
browse.pack(fill='x', padx=5, pady=5)

# mouse event
global mouse_click_no
global mouse_pos
global canvas

mouse_pos = []
mouse_click_no = 0
mouse_data = []

# csv file 읽을 때마다 새 탭 생성
def create_new_tab(notebook, file_name):
    new_tab = ttk.Frame(notebook)
    notebook.add(new_tab, text=file_name)

    notebook.select(new_tab)

# open file
def open_file():
    global file_name
    file_name = fd.askopenfilename(title='Open a File',
                                   filetypes=(('csv Files', '*.csv',), ("All FIles", "*.*")),
                                   initialdir='C:/Users/User/Downloads')

    if file_name:
        try:
            filename = r"{}".format(file_name)
            df = pd.read_csv(filename)
        except ValueError:
            label.config(text="File could not be opened")
        except FileNotFoundError:
            label.config(text="File Not Found")
        # disp_csv()

    if file_name == '':
        msgbox.showwarning('Warning', 'Select a File!')

    browse['text'] = file_name

def clear():
    for item in canvas.get_tk_widget().find_all():
        canvas.get_tk_widget().delete(item)
        canvas.get_tk_widget().pack_forget()


# 주식 가져오기
def fetch_stock_data(stock_code, start_date, end_date):
    stock_data = fdr.DataReader(stock_code, start=start_date, end=end_date)
    stock_data.reset_index()
    return stock_data

start_date = '2020-10-03'
end_date = '2023-10-03'

moadata = fetch_stock_data('288980', start_date, end_date)
emro = fetch_stock_data('058970', start_date, end_date)
exem = fetch_stock_data('205100', start_date, end_date)

# menu
menu = tk.Menu(root)

file_menu = tk.Menu(menu, tearoff=0)
file_menu.add_command(label='Open File', accelerator='Ctrl+O', command=open_file)
menu.add_cascade(label='File', menu=file_menu)

exit_menu = tk.Menu(menu, tearoff=0)
exit_menu.add_command(label='Exit', command=quit)
menu.add_cascade(label='Exit', menu=exit_menu)

# GUI config
root.config(menu=menu)
root.geometry('1280x720')
root.resizable(False, False)
root.mainloop()