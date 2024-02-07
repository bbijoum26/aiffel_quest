import tkinter.ttk as ttk
import tkinter as tk
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tkinter import *
from unidecode import unidecode
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# font
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

class MainWindow:
    def __init__(self):
        self.root = Tk()
        self.root.title('리그오브레전드 분석 프로그램')
        image = tk.PhotoImage(file='/Users/project/mj/LOL/riot_icon.png')
        self.root.iconphoto(True, image)
        self.root.geometry('1036x600')

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)

        champion_frame = ttk.Frame(self.notebook)
        self.notebook.add(champion_frame, text='Champion Info')
        self.champion_info = ChampionInfo(champion_frame, champion_csv='/Users/project/mj/LOL/csv_file/champion.csv')

        rate_frame = ttk.Frame(self.notebook)
        self.notebook.add(rate_frame, text='Pick/Win/Ban')
        self.champion_rate = ChampionRate(rate_frame, champion_csv='champion.csv')

        match_frame = ttk.Frame(self.notebook)
        self.notebook.add(match_frame, text='Winning Factor')
        self.champion_rate = WinLose(match_frame, winner_csv='match_winner.csv', loser_csv='match_loser.csv')

        object_frame = ttk.Frame(self.notebook)
        self.notebook.add(object_frame, text='Object')
        self.champion_rate = ObjectKills(object_frame, winner_csv='match_winner.csv', loser_csv='match_loser.csv')    

        self.root.mainloop()

class ChampionInfo:
    def __init__(self, root, champion_csv):
        self.root = root
        self.champion_csv = champion_csv

        self.create_widgets()

    def create_widgets(self):
        # 검색 프레임
        self.search_frame = tk.Frame(self.root)
        self.search_frame.pack(pady=10)

        # 검색 입력 필드
        self.search_entry = tk.Entry(self.search_frame, width=30)
        self.search_entry.grid(row=0, column=0, padx=5)

        # 검색 버튼
        self.search_button = tk.Button(self.search_frame, text="검색", command=self.search_champion, relief="flat", bg='#58a676', fg="white")
        self.search_button.grid(row=0, column=1, padx=5, sticky="w")

        # 이미지
        self.icon_label = tk.Label(self.root)
        self.icon_label.pack(padx=10, pady=10)

        # 이름
        self.name_label = tk.Label(self.root)
        self.name_label.pack(padx=10, pady=5)

        # 타이틀
        self.title_label = tk.Label(self.root)
        self.title_label.pack(padx=10, pady=5)

        # 태그
        self.tags_label = tk.Label(self.root)
        self.tags_label.pack(padx=10, pady=5)

        # 난이도
        self.difficulty_label = tk.Label(self.root)
        self.difficulty_label.pack(padx=10, pady=5)

        # 그래프
        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(padx=10, pady=10)

    def search_champion(self):
        champion_name = self.search_entry.get()

        # CSV 파일에서 챔피언 정보 가져오기
        champion_data = pd.read_csv(self.champion_csv)

        # 챔피언 이름에 해당하는 데이터 추출
        champion_stats = champion_data[champion_data['id'] == champion_name]

        if champion_stats.empty:
            self.clear_info()
            self.clear_graph()
            return

        # 챔피언 정보 표시
        self.display_champion_info(champion_stats)

        # 통계 그래프 표시
        self.display_statistics(champion_stats)

    def display_champion_info(self, champion_stats):
        # 이미지 표시
        champion_image = []
        datas = requests.get('https://ddragon.leagueoflegends.com/cdn/13.11.1/data/ko_KR/champion.json')
        datas = datas.json()
        
        for data in datas["data"]:
            champion_image.append(data)

        champion_image = tk.PhotoImage(file=f"{champion_stats['id'].values[0]}.png")
        self.icon_label.configure(image=champion_image)
        self.icon_label.image = champion_image

        # 이름
        self.name_label.configure(text=champion_stats['name'].values[0])

        # 타이틀
        self.title_label.configure(text=champion_stats['title'].values[0])

        # 태그
        tags = ', '.join(champion_stats['tags'].values[0].split('|'))
        self.tags_label.configure(text=tags)

        # 난이도
        difficulty = champion_stats['difficulty'].values[0]
        self.difficulty_label.configure(text=f"난이도: {difficulty}")

    def display_statistics(self, champion_stats):
        # 승률(%), 픽률(%), 밴율(%) 데이터 추출
        win_rate = champion_stats['승률(%)'].values[0]
        pick_rate = champion_stats['픽률(%)'].values[0]
        ban_rate = champion_stats['밴율(%)'].values[0]

        # 데이터를 100%를 기준으로 조정
        total = win_rate + pick_rate + ban_rate
        win_rate = win_rate
        pick_rate = pick_rate
        ban_rate = ban_rate

        # 막대 그래프 그리기
        labels = ['승률', '픽률', '밴율']
        values = [win_rate, pick_rate, ban_rate]

        plt.figure(figsize=(5, 4))
        bars = plt.barh(labels, values)
        plt.title('통계')
        plt.xlabel('항목')
        plt.ylabel('비율')
        plt.xlim(0, 100)

        # 값 표시
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height() / 2, f'{float(width)}%', ha='left', va='center')
        
        # 그래프를 Tkinter에 표시
        self.clear_graph()
        self.graph = FigureCanvasTkAgg(plt.gcf(), master=self.graph_frame)
        self.graph.get_tk_widget().pack()

    def clear_info(self):
        self.icon_label.configure(image=None)
        self.name_label.configure(text='')
        self.title_label.configure(text='')
        self.tags_label.configure(text='')
        self.difficulty_label.configure(text='')

    def clear_graph(self):
        if hasattr(self, 'graph'):
            self.graph.get_tk_widget().pack_forget()

class ChampionRate:
    def __init__(self, root, champion_csv):
        self.root = root
        self.champion_csv = champion_csv

        self.create_widgets()

    def create_widgets(self):
        # 라디오 버튼 프레임 생성
        radio_frame = tk.Frame(self.root)
        radio_frame.pack(side=tk.TOP, padx=5, pady=5)

        # 라디오 버튼 생성
        self.radio_var = tk.IntVar()
        self.radio_var.set(0)  # 초기 선택값
        radio_button1 = tk.Radiobutton(radio_frame, text="승률-픽률", variable=self.radio_var, value=0)
        radio_button2 = tk.Radiobutton(radio_frame, text="픽률-밴율", variable=self.radio_var, value=1)
        radio_button3 = tk.Radiobutton(radio_frame, text="밴율-승률", variable=self.radio_var, value=2)
        radio_button4 = tk.Radiobutton(radio_frame, text="승률-플레이수", variable=self.radio_var, value=3)
        radio_button5 = tk.Radiobutton(radio_frame, text="픽률-플레이수", variable=self.radio_var, value=4)
        radio_button6 = tk.Radiobutton(radio_frame, text="밴율-플레이수", variable=self.radio_var, value=5)

        radio_button1.pack(side=tk.LEFT, padx=5)
        radio_button2.pack(side=tk.LEFT, padx=5)
        radio_button3.pack(side=tk.LEFT, padx=5)
        radio_button4.pack(side=tk.LEFT, padx=5)
        radio_button5.pack(side=tk.LEFT, padx=5)
        radio_button6.pack(side=tk.LEFT, padx=5)

        # 버튼 생성
        self.plot_button1 = tk.Button(radio_frame, text="산점도 그리기", command=self.draw_scatter_plot)
        self.plot_button2 = tk.Button(radio_frame, text='히트맵 그리기', command=self.display_correlation_table)

        self.plot_button1.pack(side=tk.LEFT, padx=5)
        self.plot_button2.pack(side=tk.LEFT, padx=5)

        # 그래프 표시 영역
        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(padx=10, pady=10)


    def draw_scatter_plot(self):
        # 선택된 라디오 버튼 값 확인
        radio_value = self.radio_var.get()

        # 챔피언 데이터 가져오기
        champion_data = pd.read_csv(self.champion_csv)
        champion_data['플레이수'] = champion_data['플레이수'].str.replace(',', '').astype(int)

        # 선택된 항목 가져오기
        if radio_value == 0:
            x_label = '승률(%)'
            y_label = '픽률(%)'
        elif radio_value == 1:
            x_label = '픽률(%)'
            y_label = '밴율(%)'
        elif radio_value == 2:
            x_label = '승률(%)'
            y_label = '밴율(%)'
        elif radio_value == 3:
            x_label = '승률(%)'
            y_label = '플레이수'
        elif radio_value == 4:
            x_label = '픽률(%)'
            y_label = '플레이수'
        elif radio_value == 5:
            x_label = '밴율(%)'
            y_label = '플레이수'

        # 선택된 항목에 대한 산점도 그리기
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=x_label, y=y_label, data=champion_data)
        plt.title('Champion Rate Scatter Plot')
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # 그래프를 Tkinter에 표시
        self.clear_graph()
        self.graph = FigureCanvasTkAgg(plt.gcf(), master=self.graph_frame)
        self.graph.get_tk_widget().pack()
    
    def display_correlation_table(self):
        # 챔피언 데이터 가져오기
        champion_data = pd.read_csv(self.champion_csv)

        # 플레이수 숫자로 변환
        champion_data['플레이수'] = champion_data['플레이수'].str.replace(',', '').astype(int)

        # 상관 계수 표 생성
        correlation_table = champion_data[['플레이수', '승률(%)', '픽률(%)', '밴율(%)']].corr()

        # seaborn을 사용하여 히트맵 그리기
        plt.figure(figsize=(10, 5))
        sns.heatmap(correlation_table, annot=True, cmap='coolwarm')
        plt.title('Correlation Coefficient Table')

        # 그래프를 Tkinter에 표시
        self.clear_graph()
        self.graph = FigureCanvasTkAgg(plt.gcf(), master=self.graph_frame)
        self.graph.draw()
        self.graph.get_tk_widget().pack()

    def clear_graph(self):
        if hasattr(self, 'graph'):
            self.graph.get_tk_widget().pack_forget()

class WinLose:
    def __init__(self, root, winner_csv, loser_csv):
        self.root = root
        self.winner_csv = winner_csv
        self.loser_csv = loser_csv
        self.create_widgets()

    def create_widgets(self):
        # 프레임 생성
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        # 라디오 버튼 생성
        self.radio_var = tk.IntVar()
        self.radio_var.set(0)  # 초기 선택값
        radio_button1 = tk.Radiobutton(frame, text="First Blood", variable=self.radio_var, value=0)
        radio_button2 = tk.Radiobutton(frame, text="First Tower", variable=self.radio_var, value=1)
        radio_button3 = tk.Radiobutton(frame, text="First Inhibitor", variable=self.radio_var, value=2)
        radio_button4 = tk.Radiobutton(frame, text="First Baron", variable=self.radio_var, value=3)
        radio_button5 = tk.Radiobutton(frame, text="First Dragon", variable=self.radio_var, value=4)
        radio_button6 = tk.Radiobutton(frame, text="First Rift Herald", variable=self.radio_var, value=5)

        radio_button1.grid(row=0, column=0, sticky=tk.W)
        radio_button2.grid(row=0, column=1, sticky=tk.W)
        radio_button3.grid(row=0, column=2, sticky=tk.W)
        radio_button4.grid(row=0, column=3, sticky=tk.W)
        radio_button5.grid(row=0, column=4, sticky=tk.W)
        radio_button6.grid(row=0, column=5, sticky=tk.W)
            
        # 그래프 표시 영역
        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(padx=10, pady=10)

        # 버튼 생성
        plot_button = tk.Button(frame, text="그래프 그리기", command=self.plot_pie_chart)
        plot_button.grid(row=0, column=6, padx=5)

    def plot_pie_chart(self):
        # 선택된 라디오 버튼 값 확인
        radio_value = self.radio_var.get()

        # CSV 파일 읽기
        win_df = pd.read_csv(self.winner_csv)
        lose_df = pd.read_csv(self.loser_csv)

        # 라디오 버튼 값에 따라 선택된 컬럼 가져오기
        column_names = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald']
        selected_column = column_names[radio_value]

        # 선택된 컬럼 값에 따른 'win'과 'lose'의 개수 구하기
        win_count = win_df[win_df[selected_column] == True][selected_column].count()
        lose_count = lose_df[lose_df[selected_column] == True][selected_column].count()

        # 파이 그래프 그리기
        labels = ['Win', 'Lose']
        sizes = [win_count, lose_count]
        colors = ['lightblue', 'lightcoral']
        explode = (0.1, 0)  # 돌출 효과 설정

        fig = plt.figure(figsize=(10, 5))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')  # 원형으로 표시
        plt.title(f'Win/Lose Ratio based on {selected_column}')

        # 그래프를 표시할 캔버스 생성
        self.clear_graph()
        self.graph = FigureCanvasTkAgg(plt.gcf(), master=self.graph_frame)
        self.graph.draw()
        self.graph.get_tk_widget().pack()

    def clear_graph(self):
        if hasattr(self, 'graph'):
            self.graph.get_tk_widget().pack_forget()

class ObjectKills:
    def __init__(self, root, winner_csv, loser_csv):
        self.root = root
        self.winner_csv = winner_csv
        self.loser_csv = loser_csv
        self.create_widgets()

    def create_widgets(self):
        # 프레임 생성
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        # 라디오 버튼 생성
        self.radio_var = tk.IntVar()
        self.radio_var.set(0)  # 초기 선택값
        radio_button1 = tk.Radiobutton(frame, text="Tower", variable=self.radio_var, value=0)
        radio_button2 = tk.Radiobutton(frame, text="Inhibitor", variable=self.radio_var, value=1)
        radio_button3 = tk.Radiobutton(frame, text="Baron", variable=self.radio_var, value=2)
        radio_button4 = tk.Radiobutton(frame, text="Dragon", variable=self.radio_var, value=3)
      
        radio_button1.grid(row=0, column=0, sticky=tk.W)
        radio_button2.grid(row=0, column=1, sticky=tk.W)
        radio_button3.grid(row=0, column=2, sticky=tk.W)
        radio_button4.grid(row=0, column=3, sticky=tk.W)
            
        # 그래프 표시 영역
        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(padx=10, pady=10)

        # 버튼 생성
        plot_button = tk.Button(frame, text="그래프 그리기", command=self.plot_bar_chart)
        plot_button.grid(row=0, column=4, padx=5)

    def plot_bar_chart(self):
        # 선택된 라디오 버튼 값 확인
        radio_value = self.radio_var.get()

        # CSV 파일 읽기
        win_df = pd.read_csv(self.winner_csv)
        lose_df = pd.read_csv(self.loser_csv)

        # 라디오 버튼 값에 따라 선택된 컬럼 가져오기
        column_names = ['towerKills', 'inhibitorKills', 'baronKills', 'dragonKills']
        selected_column = column_names[radio_value]

        # 선택된 컬럼 값에 따른 'win'과 'lose'의 개수 구하기
        win_count = win_df[win_df[selected_column] == True].shape[0]
        lose_count = lose_df[lose_df[selected_column] == True].shape[0]

        # 막대 그래프 그리기
        labels = ['Win', 'Lose']
        values = [win_count, lose_count]

        plt.figure(figsize=(10, 5))
        plt.barh(labels, values, color=['lightblue', 'lightcoral'])
        plt.title(f'Win/Lose Count based on {selected_column}')
        plt.xlabel('Outcome')
        plt.ylabel('Count')

        # 그래프를 표시할 캔버스 생성
        self.clear_graph()
        self.graph = FigureCanvasTkAgg(plt.gcf(), master=self.graph_frame)
        self.graph.draw()
        self.graph.get_tk_widget().pack()
    
    def clear_graph(self):
        if hasattr(self, 'graph'):
            self.graph.get_tk_widget().pack_forget()


if __name__ == '__main__':
    app = MainWindow()
    app.root.mainloop()
