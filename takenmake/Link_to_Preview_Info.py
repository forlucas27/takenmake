import requests
from bs4 import BeautifulSoup


def get_preview_info(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    #title = soup.title.text
    try:
        image = soup.select('body img')[0]['src']  # 'data-lazy-src'
        r = requests.head(image)
    except:
        try:
            image = soup.select('div img')[0]['data-lazy-src']
            r = requests.head(image)
        except:
            image = soup.select('picture img')[0]['data-src']

    if 'logo' in image:
        try:
            image = soup.select('div img')[2]['data-src']  # 'data-lazy-src'
            r = requests.head(image)
        except:
            try:
                image = soup.select('div img')[2]['data-lazy-src']
                r = requests.head(image)
            except:
                image = soup.select('picture img')[2]['src']

    return image
