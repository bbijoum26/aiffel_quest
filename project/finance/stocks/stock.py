# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
import seaborn as sns
from pykrx import stock
import pandas_datareader.data as pdr
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# GUI import
import tkinter as tk
import tkinter.ttk
import tkinter.messagebox as msgbox
from tkinter import ttk
from tkinter import *
from tksheet import Sheet
from tkinter import filedialog as fd

# ML import
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

global canvas

# tkinter
root = tk.Tk()
root.title('Stock Price Prediction')
notebook = tkinter.ttk.Notebook(root, width=1280, height=600)
notebook.pack()

label = Label(root)
label.pack()

# 그래프 지우기
def clear():
    for item in canvas.get_tk_widget().find_all():
        canvas.get_tk_widget().delete(item)
        canvas.get_tk_widget().pack_forget()



###### Stock Data ######

# CSV 파일에서 주식 데이터 가져오기
def fetch_stock_data_from_csv(csv_file_path):
    stock_data = pd.read_csv(csv_file_path)
    return stock_data

# 주식 데이터 가져오기
def fetch_stock_data(stock_code, start_date, end_date):
    if isinstance(stock_code, int) or stock_code.isdigit():
        # 입력된 stock_code가 숫자일 경우 yfinance를 사용하여 데이터 가져오기
        stock_data = fdr.DataReader(stock_code, start=start_date, end=end_date)
    elif isinstance(stock_code, str):
        # 입력된 stock_code가 문자열일 경우 FinanceDataReader를 사용하여 데이터 가져오기
        stock_data = yf.download(str(stock_code), start=start_date, end=end_date)
    else:
        raise ValueError("Invalid stock_code format")

    stock_data = stock_data.reset_index()
    return stock_data


# 주식 데이터를 Treeview에 추가하는 함수
def add_stock_data_to_treeview(stock_data):
    tree.delete(*tree.get_children())  # 기존 데이터 삭제

    # stock_data의 칼럼 이름을 사용하여 Treeview에 데이터 추가
    for _, row in stock_data.iterrows():
        values = tuple(row)
        tree.insert('', 'end', values=values)

# 파일 열기 버튼을 클릭할 때 호출되는 함수
def open_file():
    global file_name
    file_name = fd.askopenfilename(title='Open a File',
                                   filetypes=(("csv Files", "*.csv"), ("All Files", "*.*")))
    
    if file_name:
        try:
            stock_data = fetch_stock_data_from_csv(file_name)
            add_stock_data_to_treeview(stock_data)  # 수정된 함수로 데이터를 Treeview에 추가
        except ValueError:
            label.config(text='File Could not be append')
        except FileNotFoundError:
            label.config(text='File Not Found')
            
    if file_name == '':
        msgbox.showwarning("Warning", 'Select a File!')

# 주식 데이터를 불러와 Treeview에 표시하는 함수
def display_stock_data():
    global tree 
    
    stock_code = stock_code_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    
    if not stock_code or not start_date or not end_date:
        return  # 입력값이 없으면 아무것도 하지 않음

    try:
        stock_data = fetch_stock_data(stock_code, start_date, end_date)
        
        # Treeview에 데이터 추가하기 전에 기존 데이터 삭제
        tree.delete(*tree.get_children())
        
        # Treeview 칼럼 정의
        if not tree['columns']:
            columns = stock_data.columns
            tree['columns'] = tuple(columns)
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)  # 칼럼 너비 조정

        # Treeview에 데이터 추가
        for index, row in stock_data.iterrows():
            values = tuple(row)
            tree.insert('', 'end', values=values)
    except Exception as e:
        print(str(e))  # 오류 메시지 출력
        

def clear_data():
    global stock_data
    
    stock_data = None
    tree.delete(*tree.get_children())
    label.config(text='Data cleared')


# stock tab
stock_tab = tkinter.Frame(root)
notebook.add(stock_tab, text='Stock Data')

stock_frame = LabelFrame(stock_tab, text='Stock Data')
stock_frame.pack(fill='y', side='left')

stock_visulization = Frame(stock_tab)
stock_visulization.pack()

# stock code entry widget
stock_code_label = tk.Label(stock_frame, text='Stock Code:')
stock_code_label.pack(fill='both', padx=5, pady=5)
stock_code_entry = tk.Entry(stock_frame)
stock_code_entry.pack(padx=5, pady=5)

start_date_label = tk.Label(stock_frame, text='Start Date (YYYY-MM-DD):')
start_date_label.pack(fill='both', padx=5, pady=5)
start_date_entry = tk.Entry(stock_frame)
start_date_entry.pack(fill='both', padx=5, pady=5)

end_date_label = tk.Label(stock_frame, text='End Date (YYYY-MM-DD):')
end_date_label.pack(padx=5, pady=5)
end_date_entry = tk.Entry(stock_frame)
end_date_entry.pack(padx=5, pady=5)

fetch_button = tk.Button(stock_frame, text='Sheet', width=10, command=display_stock_data)
fetch_button.pack(padx=5, pady=10)

clear_button = tk.Button(stock_frame, text='Clear', width=10, command=clear_data)
clear_button.pack(padx=5, pady=10)

# Treeview 생성
tree = ttk.Treeview(stock_visulization, height=800)
tree.pack(fill='both', expand=True)


# # Treeview 생성(고정 columns)
# tree = ttk.Treeview(stock_visulization, columns=('Open', 'High', 'Low', 'Close', 'Volume', 'Change'),height=800)
# tree.heading('#1', text='Open')
# tree.heading('#2', text='High')
# tree.heading('#3', text='Low')
# tree.heading('#4', text='Close')
# tree.heading('#5', text='Volume')
# tree.heading('#6', text='Change')

# tree.column('#0', width=20)

# # Treeview 칼럼 크기 설정
# for i, col_name in enumerate(['Open', 'High', 'Low', 'Close', 'Volume', 'Change']):
#     tree.column(f'#{i+1}', width=165, anchor='w')

# tree.pack(fill='both', expand=True)

menu = tk.Menu(root)

file_menu = tk.Menu(menu, tearoff=0)
file_menu.add_command(label='Open File', accelerator='command+O', command=open_file)
menu.add_cascade(label='File', menu=file_menu)

exit_menu = tk.Menu(menu, tearoff=0)
exit_menu.add_command(label='Exit', command=quit)
menu.add_cascade(label='Exit', menu=exit_menu)



###### Visualization ######

# stock price
def plot_stock_price(stock_data):
    global canvas
    f = plt.figure(figsize=(16, 8))
    plt.title('Stock Price Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(stock_data['High'], label='High Price')
    plt.plot(stock_data['Low'], label='Low Price')
    plt.legend()
    
    canvas = FigureCanvasTkAgg(f, canvas_visulization)
    canvas.draw()
    canvas.get_tk_widget().pack()

def fetch_and_plot_stock_price():
    stock_code = stock_code_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    if not stock_code or not start_date or not end_date:
        msgbox.showerror('Error', 'Please enter valid stock code and date range')
        return

    try:
        stock_data = fetch_stock_data(stock_code, start_date, end_date)
        plot_stock_price(stock_data)
    except Exception as e:
        msgbox.showerror('Error', str(e))
 
# stock returns
def plot_stock_return(stock_data):
    global canvas
    
    f = plt.figure(figsize=(16, 8))
    plt.title('Stock Returns Distribution')
    plt.xlabel('Rate of Return')
    plt.ylabel('Distribution')
    plt.hist(stock_data['Change'], density=True, bins=300)
    plt.legend()
    
    canvas = FigureCanvasTkAgg(f, canvas_visulization)
    canvas.draw()
    canvas.get_tk_widget().pack()

def fetch_and_plot_stock_return():
    stock_code = stock_code_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    if not stock_code or not start_date or not end_date:
        msgbox.showerror('Error', 'Please enter valid stock code and date range')
        return

    try:
        stock_data = fetch_stock_data(stock_code, start_date, end_date)
        plot_stock_return(stock_data)
    except Exception as e:
        msgbox.showerror('Error', str(e))

# stock close
def plot_stock_close(stock_data):
    global canvas
    
    f = plt.figure(figsize=(16, 8))
    plt.title('Stock Closinf Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.plot(stock_data['Close'])
    plt.legend()
    
    canvas = FigureCanvasTkAgg(f, canvas_visulization)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
def fetch_and_plot_stock_close():
    stock_code = stock_code_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()

    if not stock_code or not start_date or not end_date:
        msgbox.showerror('Error', 'Please enter valid stock code and date range')
        return

    try:
        stock_data = fetch_stock_data(stock_code, start_date, end_date)
        plot_stock_close(stock_data)
    except Exception as e:
        msgbox.showerror('Error', str(e))

# visulization tab
visualization_tab = tkinter.Frame(root)
notebook.add(visualization_tab, text='Visulization')

visualization_frame = LabelFrame(visualization_tab, text='Visulization')
visualization_frame.pack(fill='y', side='left')

canvas_visulization = Frame(visualization_tab)
canvas_visulization.pack(fill='both')

# visulization button
price_button = tk.Button(visualization_frame, text='Price', width=10, command=fetch_and_plot_stock_price)
price_button.pack(padx=5, pady=10)

return_button = tk.Button(visualization_frame, text='Rate of Returns', width=10, command=fetch_and_plot_stock_return)
return_button.pack(padx=5, pady=10)

close_button = tk.Button(visualization_frame, text='Close', width=10, command=fetch_and_plot_stock_close)
close_button.pack(padx=5, pady=10)

btn_clear = tk.Button(visualization_frame, text="Clear", width=10, command=clear)
btn_clear.pack(padx=5, pady=10)


###### Data Preprocessing ######

# data preprocessing
def data_preprocessing(stock_data):
    # 주식 데이터를 날짜 형식으로 변환
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d')
    
    # 컬럼 이름을 소문자 및 공백 대신 언더스코어(_)로 변경
    stock_data.columns = [str(x).lower().replace(' ', '_') for x in stock_data.columns]
    
    # 날짜를 기준으로 오름차순 정렬
    stock_data.sort_values(by='date', inplace=True, ascending=True)
    
    # y 변수 선택 (종가를 예측하려면 'close' 컬럼을 선택)
    stock_ = stock_data[['close']]
    
    # X 변수 선택 (종가를 예측하기 위한 특성 선택)
    forecast_out = 10  # 종가를 10일 후로 예측
    stock_['forecast'] = stock_['close'].shift(-forecast_out)
    
    # NAN 데이터 제거
    stock_ = stock_.dropna()
    
    return stock_

# Today's vs Previous day's
def plot_today_previous(stock_):
    global canvas
    
    f = plt.figure(figsize=(16, 8))
    plt.title("Today's closing price VS Previous day's closing price")
    plt.plot(stock_['close'], color='blue', label="Today's Close")
    plt.plot(stock_['forecast'], color='red', label="Previous Day's Close")
    plt.xlabel('Date')
    plt.ylabel('Closing Price KRW')
    plt.legend()
  
    canvas = FigureCanvasTkAgg(f, canvas_preprocessing)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
def fetch_and_plot_today_previous():
    stock_code = stock_code_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    
    if not stock_code or not start_date or not end_date:
        msgbox.showerror('Error', 'Please enter valid stock code and date range')
        return

    try:
        stock_data = fetch_stock_data(stock_code, start_date, end_date)
        stock_ = data_preprocessing(stock_data)
        plot_today_previous(stock_)
    except Exception as e:
        msgbox.showerror('Error', str(e))

# Moving Average 
def plot_moving_average(stock_):
    global canvas
    
    f = plt.figure(figsize=(16, 8))
    
    stock_['MA10'] = stock_.close.rolling(10).mean() # 10일전 평균 데이터
    
    plt.title("Closing Price Time Series")
    plt.plot(stock_['close'], color='blue', label="Today")
    plt.plot(stock_['MA10'], color='red', label="10 days ago")
    plt.xlabel('Date')
    plt.ylabel('Closing Price KRW')
    plt.legend()
  
    canvas = FigureCanvasTkAgg(f, canvas_preprocessing)
    canvas.draw()
    canvas.get_tk_widget().pack()

def fetch_and_plot_moving_average():
    stock_code = stock_code_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    
    if not stock_code or not start_date or not end_date:
        msgbox.showerror('Error', 'Please enter valid stock code and date range')
        return

    try:
        stock_data = fetch_stock_data(stock_code, start_date, end_date)
        stock_ = data_preprocessing(stock_data)
        plot_moving_average(stock_)
    except Exception as e:
        msgbox.showerror('Error', str(e))

# Today VS 10 days age
def plot_today_ten(stock_):
    global canvas
    
    stock_code = stock_code_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    stock_data = fetch_stock_data(stock_code, start_date, end_date)
    
    stock_['MA10'] = stock_.close.rolling(10).mean() # 10일전 평균 데이터
    f = plt.figure(figsize=(16, 8))
    plt.title("Today's closing price VS Closing price 10 days ago")
    plt.plot(stock_data['Close'], color='blue', label="Today")
    plt.plot(stock_['MA10'], color='red', label="10 days ago")
    plt.xlabel('Date')
    plt.ylabel('Closing Price KRW')
    plt.legend()
  
    canvas = FigureCanvasTkAgg(f, canvas_preprocessing)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
def fetch_and_plot_today_ten():
    stock_code = stock_code_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    
    if not stock_code or not start_date or not end_date:
        msgbox.showerror('Error', 'Please enter valid stock code and date range')
        return

    try:
        stock_data = fetch_stock_data(stock_code, start_date, end_date)
        stock_ = data_preprocessing(stock_data)
        plot_today_ten(stock_)
    except Exception as e:
        msgbox.showerror('Error', str(e))

# preprocessing tab
preprocessing_tab = tkinter.Frame(root)
notebook.add(preprocessing_tab, text='Preprocessing')

preprocessing_frame = LabelFrame(preprocessing_tab, text='Visualization')
preprocessing_frame.pack(fill='y', side='left')

canvas_preprocessing = Frame(preprocessing_tab)
canvas_preprocessing.pack(fill='both')

# visualization button
tp_button = tk.Button(preprocessing_frame, text='Today/Previous', width=10, command=fetch_and_plot_today_previous)
tp_button.pack(padx=5, pady=10)

mv_button = tk.Button(preprocessing_frame, text='10 Days age', width=10, command=fetch_and_plot_moving_average)
mv_button.pack(padx=5, pady=10)

tt_button = tk.Button(preprocessing_frame, text='Today/Ten', width=10, command=fetch_and_plot_today_ten)
tt_button.pack(padx=5, pady=10)

btn_clear1 = tk.Button(preprocessing_frame, text='Clear', width=10, command=clear)
btn_clear1.pack(padx=5, pady=10)



###### Modeling ######

# X, y data
def train_test_data():
    stock_code = stock_code_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    
    stock_data = fetch_stock_data(stock_code, start_date, end_date)
    stock_ = data_preprocessing(stock_data)
    
    X = np.array(stock_.drop(columns='forecast'))
    y = np.array(stock_['forecast'])
    
    X_train, X_test, y_tratin, y_test = train_test_split(X, y, test_size=0.3)
    
    return X_train, X_test, y_tratin, y_test 

# 모델 예측 및 평가
def confusion_model(model, X_test, y_test):
    global canvas
    
    # 모델 예측
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    f = plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    canvas = FigureCanvasTkAgg(f, canvas_modelling)
    canvas.draw()
    canvas.get_tk_widget().pack()


# RandomForest
def randomforest():
    X_train, X_test, y_train, y_test = train_test_data()
    
    # RandomForest 모델 생성 및 학습
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    score_label = Label(score_frame, text=f'Score: {model.score(X_train, y_train)}')
    score_label.pack()
    
    # 모델 평가 및 결과 출력
    confusion_model(model, X_test, y_test)

# LogisticRegression
def logistic():
    X_train, X_test, y_train, y_test = train_test_data()
    
    # 모델 생성 및 학습
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    score_label = Label(score_frame, text=f'Score: {model.score(X_train, y_train)}')
    score_label.pack()
    
    # 모델 평가 및 결과 출력
    confusion_model(model, X_test, y_test)

# SVM
def svm():
    X_train, X_test, y_train, y_test = train_test_data()
    
    # 모델 생성 및 학습
    model = SVC()
    model.fit(X_train, y_train)
    
    score_label = Label(score_frame, text=f'Score: {model.score(X_train, y_train)}')
    score_label.pack()
    
    # 모델 평가 및 결과 출력
    confusion_model(model, X_test, y_test)

# naive bayes
def naive_bayes():
    X_train, X_test, y_train, y_test = train_test_data()
    
    # 모델 생성 및 학습
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    score_label = Label(score_frame, text=f'Score: {model.score(X_train, y_train)}')
    score_label.pack()
    
    # 모델 평가 및 결과 출력
    confusion_model(model, X_test, y_test)

    
# modelling tab
modelling_tab = tkinter.Frame(root)
notebook.add(modelling_tab, text='Modelling')

modelling_frame = LabelFrame(modelling_tab, text='Machine Learning')
modelling_frame.pack(fill='y', side='left')

score_frame = LabelFrame(modelling_tab, text='Score')
score_frame.pack(fill='both', side='bottom')

canvas_modelling = Frame(modelling_tab)
canvas_modelling.pack(fill='both')

# ML button
random_button = tk.Button(modelling_frame, text='RandomForest', width=10, command=randomforest)
random_button.pack(padx=5, pady=10)

logistic_button = tk.Button(modelling_frame, text='Logistic', width=10, command=logistic)
logistic_button.pack(padx=5, pady=10)

svm_button = tk.Button(modelling_frame, text='SVM', width=10, command=svm)
svm_button.pack(padx=5, pady=10)

naive_button = tk.Button(modelling_frame, text='Navie Bayes', width=10, command=naive_bayes)
naive_button.pack(padx=5, pady=10)

btn_clear2 = tk.Button(modelling_frame, text='Clear', width=10, command=clear)
btn_clear2.pack(padx=5, pady=10)


root.config(menu=menu)
root.geometry('1280x720')
root.mainloop()