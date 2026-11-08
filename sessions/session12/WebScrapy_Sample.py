# Session 12 — scrape Dior products from liliome.com into a Pandas table.
# Writes lilium_df.csv and optional perfume/ images next to this file.
# Needs a network connection. From the repo root:
#   uv run python sessions/session12/WebScrapy_Sample.py
#
# region Import Libraries
# pip install requests bs4 selenium pandas

from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd
import selenium

_SESSION = Path(__file__).resolve().parent
_PERFUME = _SESSION / "perfume"
_PERFUME.mkdir(exist_ok=True)

# endregion

# region requests ...


url_home_page: str = "https://liliome.com/brand/dior/"
page_number = 1

response = requests.get(url_home_page)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    ul_tag = soup.find("ul", {"class": "page-numbers nav-pagination links text-center"})
    count_li = len(ul_tag.find_all('li'))
    page_number = ul_tag.find_all('li')[-2].text
    # print(page_number)

persian_title_list = []
english_title_list = []
old_price_list = []
new_price_list = []
rate_list = []
image_list = []

for item in range(1, int(page_number) + 1):

    site_url = f"https://liliome.com/brand/dior/page/{item}/"
    response = requests.get(site_url)
    # print(response.text) # HTML
    # print(response.status_code)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Container BOX
        container_div = soup.find("div", {
            "class": 'products row row-small large-columns-5 medium-columns-4 small-columns-2 has-shadow row-box-shadow-2 row-box-shadow-3-hover has-equal-box-heights equalize-box'})
        # print(container_div.text)
        small_boxes = container_div.find_all("div", {"class": "product-small box"})
        # print(len(small_boxes))
        for small_box in small_boxes:
            title = small_box.find("p", {"class": "name product-title woocommerce-loop-product__title"}).text
            # print(title)
            persian_title_list.append(title.split('|')[0])
            english_title_list.append(title.split('|')[1])

            try:
                old_price = small_box.find_all("span", {"class": "woocommerce-Price-amount amount"})[0].text
                old_price_list.append(int(old_price.replace(',', '').replace('\xa0', '').replace('تومان', '')))
            except Exception as e:
                old_price_list.append(None)

            try:
                new_price = small_box.find_all("span", {"class": "woocommerce-Price-amount amount"})[1].text
                new_price_list.append(int(new_price.replace(',', '').replace('\xa0', '').replace('تومان', '')))
            except Exception as e:
                new_price_list.append(None)

            try:
                rate = small_box.find('strong', {"class": "rating"}).text
                rate_list.append(float(rate))
            except Exception as e:
                rate_list.append(None)

            try:
                image = small_box.find("img")
                name_image = image["src"].split('/')[-1]
                image_list.append(name_image)

                if (".jpg" or ".png" or ".jpeg" or ".webp" or ".gif") in image["src"]:
                    with open(_PERFUME / f"{name_image}.jpg", mode="wb") as f:
                        f.write(requests.get(image["src"]).content)

            except Exception as e:
                image_list.append(None)
# DataFrame
# Pandas => Series, DataFrame

# students = {
#     "Name": ["Erfan", "Ali", "Sajad", "Ehsan"],
#     "Age": [20, 30, 24, 18]
# }
# # print(students["Age"][-1])
#
# df = pd.DataFrame(students)
# print(df)

lilium_dict = {
    "persian_title": persian_title_list,
    "english_title": english_title_list,
    "old_price": old_price_list,
    "new_price": new_price_list,
    "rate_list": rate_list,
    "image_list": image_list
}

lilium_df = pd.DataFrame(lilium_dict)
print(lilium_df.to_string())
lilium_df.to_csv(_SESSION / "lilium_df.csv")

# endregion
