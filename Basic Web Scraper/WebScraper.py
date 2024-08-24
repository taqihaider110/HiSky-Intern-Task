import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'http://books.toscrape.com/'

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    books = soup.find_all('article', class_='product_pod')
    
    data = []

    for book in books:
        title = book.h3.a['title']
        price = book.find('p', class_='price_color').get_text()
        data.append({'TITLE': title, 'PRICE': price})

    df = pd.DataFrame(data)

    df.to_csv('BOOKS_DATA.csv', index=False)

    print('Data has been successfully scraped and saved to books.csv')
else:
    print('Failed to retrieve the webpage. Status code:', response.status_code)
