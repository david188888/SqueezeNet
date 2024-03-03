import requests
import os 
from bs4 import BeautifulSoup
import chardet
import re

base_url = 'https://ent.sina.com.cn/ku/star_search_index.d.html'
page = 1
headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
}
names = []
photos = []
while True:
    url = f"{base_url}?page={page}"
    response = requests.get(url, headers=headers)
    encodings = chardet.detect(response.content)['encoding']
    response.encoding = encodings
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser', from_encoding='utf-8')
        people_class = soup.find('ul', {'class': 'tv-list star_list clearfix'})
        
        if people_class is not None:

            os.makedirs('people_images', exist_ok=True)
        
            for row in people_class.find_all('li')[0:]:
                print(type(row))
                data = row.select('a > img')
                name = data[0]['alt']
                names.append(name)
                photo = data[0]['src']
            
                pattern = re.compile(r'//')
                photo = pattern.sub('https://', photo)
                
                
                photo_response = requests.get(photo,headers=headers)
                if photo_response.status_code == 200:
                    image_filename = os.path.join('people_images', f"{name}.jpg")
                    with open(image_filename, 'wb') as f:
                        f.write(photo_response.content)
                photos.append(photo)
                page += 1
        else:
            print(f"page {page} does not exist")
            break
        
    else:
        break  

        

print(f"the length of names is {len(names)}")
print(f"the length of photos is {len(photos)}")
    
with open('people.txt', 'w', encoding='utf-8') as f:
    for name in names:
        f.write(f"{name}\n")
                
        
        
    
            