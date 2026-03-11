import requests
from bs4 import BeautifulSoup
import re

def main():
    try:
        req = requests.get("https://grabcad.com/library")
        soup = BeautifulSoup(req.text, 'html.parser')
        
        # Look for links or options containing "categories="
        categories = set()
        
        # Method 1: looking at hrefs
        for link in soup.find_all('a', href=True):
            match = re.search(r'categories=([a-zA-Z0-9-]+)', link['href'])
            if match:
                categories.add(match.group(1))
                
        # Method 2: looking for checkbox/inputs or select options
        for inp in soup.find_all('input', {'name': 'categories'}):
            if inp.get('value'):
                categories.add(inp.get('value'))
                
        print("Categories found:")
        for c in sorted(list(categories)):
            print("-", c)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
