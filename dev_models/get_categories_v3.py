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
        html = driver.execute_script("return document.body.innerHTML")
        import re
        cats = re.findall(r'categories=([a-zA-Z0-9-]+)', html)
        print("Categories found via regex:")
        print(set(cats))
        
        # Another check via typical class names
        labels = driver.find_elements(By.CSS_SELECTOR, "label.checkbox")
        for l in labels:
            print("Label:", l.text)
            
    except Exception as e:
        print(f"Error: {e}")
        
    driver.quit()

if __name__ == "__main__":
    main()
