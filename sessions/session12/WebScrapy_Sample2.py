# Session 12 — Selenium browser automation (Google search demo; Digikala commented).
# Needs Chrome + a local chromedriver.exe (not stored in the repo).
# From the repo root (with chromedriver next to this file or on PATH):
#   uv run python sessions/session12/WebScrapy_Sample2.py
#
# .jpeg .jpg .gif .webp

# import requests
# from bs4 import BeautifulSoup
#
# site:str = "https://www.varzesh3.com/"
#
# # python:
# headers = {'User-Agent': 'Mozilla/5.0'}
#
# response = requests.get(site, headers=headers)
# # print(response.status_code)
#
# if response.status_code == 200:
#     # print(response.text)
#     bs = BeautifulSoup(response.content, 'html.parser')
#     result = bs.find_all("img")
#     # print(len(result))
#
#     for id, img in enumerate(result,1):
#         # print(img['src'])
#         # img_url = img['src']
#         if (".jpg" or ".png" or ".jpeg" or ".webp" or ".gif") in img["src"]:
#             with open(f"varzesh3/{id}.jpg", mode="wb") as f:
#                 f.write(requests.get(img["src"]).content)
#

# VPNLY => Chrome Extension
# pip install selenium webdriver_manager

# https://googlechromelabs.github.io/chrome-for-testing/

from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

url:str = "https://www.digikala.com/"
url_google:str = "https://www.google.com/"

# Browser: Google Chrome => Chrome Driver
# Place chromedriver.exe next to this file (not committed to the repo).

_SESSION = Path(__file__).resolve().parent
chrome_driver_path = str(_SESSION / "chromedriver.exe")
service = Service(chrome_driver_path)

driver = webdriver.Chrome(service=service)
# driver.get(url)
driver.get(url_google)
time.sleep(2)
driver.maximize_window()


# Google
'''//*[@id="ti6dpd"]'''

search_box = driver.find_element(By.XPATH, '//*[@id="ti6dpd"]')
# search_box.click()
search_box.send_keys("چکش تخریب رونیکس")
search_box.send_keys(Keys.ENTER)

time.sleep(5)




# Digikala
# -----------------------------------------------------------------------------------------------
# search_box = driver.find_element(By.XPATH, '//*[@id="base_layout_desktop_fixed_header"]/header/div[2]/div/div/div[1]/div/div/div/div/span[1]/label/div')
# search_box.click()
#
# time.sleep(5)
# '''چکش تخریب رونیکس'''
#
# search_box_input = driver.find_element(By.XPATH, '/html/body/dk-teleport[2]/div/div/div[1]/div/div/span/label/div/div/input')
# search_box_input.send_keys("چکش تخریب رونیکس")
# search_box_input.send_keys(Keys.ENTER)
# -----------------------------------------------------------------------------------------------

time.sleep(100)














