from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import warnings
from bs4 import BeautifulSoup
import pandas as pd

chrome_options = Options()
chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"]) # 셀레니움 로그 무시
warnings.filterwarnings("ignore", category=DeprecationWarning) # Deprecated warning 무시 

url = "https://www.op.gg/statistics/champions"

driver = webdriver.Chrome('c:/chromedriver.exe', options=chrome_options)
driver.get(url)  # 해당 URL을 기동시킴

soup = BeautifulSoup(driver.page_source, 'lxml')

champion_list = []
winner_rate = []
pick_rate = []
ban_rate = []
play_count = []

champs = soup.select('#content-container > div > table > tbody > tr > td.css-g2y47x.e1alsbyt8 > a > strong')
winners =soup.select('#content-container > div:nth-child(2) > table > tbody > tr > td:nth-child(5)')
picks = soup.select('#content-container > div:nth-child(2) > table > tbody > tr > td:nth-child(6) > div > div.css-12go93c.e1alsbyt0')
bans = soup.select('#content-container > div:nth-child(2) > table > tbody > tr > td:nth-child(7)')
plays = soup.select('#content-container > div:nth-child(2) > table > tbody > tr > td.css-16dwbfw.e1alsbyt3')

for champ in champs:
    champion_list.append(champ.text)

for winner in winners:
    winner_rate.append(winner.text.strip('%'))

for pick in picks:
    pick_rate.append(pick.text.strip('%'))

for ban in bans:
    ban_rate.append(ban.text.strip('%'))

for play in plays:
    play_count.append(play.text)

# 데이터프레임 생성
data = {'챔피언' : champion_list,
        '승률(%)' : winner_rate,
        '픽률(%)' : pick_rate,
        '밴율(%)' : ban_rate,
        '플레이수' : play_count
}

riot_df = pd.DataFrame(data)

# 데이터프레임 파일로 저장
riot_df.to_csv('champion_statistics.csv', index=False, encoding='utf-8')

