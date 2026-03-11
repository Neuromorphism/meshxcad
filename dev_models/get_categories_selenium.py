import time
from selenium import webdriver
from selenium.webdriver.common.by import By

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    
    driver.get("https://grabcad.com/library")
    time.sleep(3)
    
    try:
        categories = driver.find_elements(By.CSS_SELECTOR, "a[href*='categories=']")
        s = set()
        for c in categories:
            href = c.get_attribute("href")
            if 'categories=' in href:
                cat = href.split('categories=')[1].split('&')[0]
                s.add(cat)
                print(f"[{cat}] {c.text}")
    except Exception as e:
        print(f"Error: {e}")
        
    driver.quit()

if __name__ == "__main__":
    main()
