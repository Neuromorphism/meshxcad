import sys
import time
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By

def main():
    download_dir = os.path.abspath("downloads")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print(f"Downloads will be saved to: {download_dir}")

    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_dir,
             "download.prompt_for_download": False,
             "directory_upgrade": True}
    options.add_experimental_option("prefs", prefs)

    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"Failed to start Chrome: {e}\nTrying Edge...")
        try:
            edge_options = webdriver.EdgeOptions()
            edge_options.add_experimental_option("prefs", prefs)
            driver = webdriver.Edge(options=edge_options)
        except Exception as e2:
            print(f"Failed to start Edge: {e2}\nPlease ensure you have a standard browser installed.")
            return

    driver.get("https://grabcad.com/library")

    print("==========================================================", flush=True)
    print("A browser window has been opened.", flush=True)
    print("Please log in to GrabCAD manually in that window.", flush=True)
    print("If you are already logged in, just verify the library page is loaded.", flush=True)
    print("Press Enter in THIS terminal when you are ready...", flush=True)
    print("==========================================================", flush=True)
    sys.stdin.readline()

    # Get categories using javascript
    print("Extracting categories...", flush=True)
    try:
        # Wait for page to fully load
        time.sleep(2)
        # Look for category options/links in the DOM
        categories = set()
        
        # In grabcad, the filter menu has elements like a[href*='categories='] or similar data attributes
        # We can also just search the HTML
        html = driver.page_source
        
        # Regex to find standard categories like categories=aviation
        found_categories = re.findall(r'categories=([a-zA-Z0-9-]+)', html)
        if found_categories:
            for c in found_categories:
                categories.add(c)
                
        # Also let's just inject the known list in case the regex misses some due to dynamic loading
        known_categories = ['3d-printing', 'aerospace', 'agriculture', 'architecture', 
                           'automotive', 'aviation', 'components', 'computer', 'construction', 
                           'design', 'educational', 'electrical', 'fixtures', 'furniture', 
                           'hobby', 'household', 'industrial-design', 'interior-design', 
                           'jewelry', 'machine-design', 'medical', 'military', 'miscellaneous', 
                           'piping', 'robotics', 'sporting-goods', 'tech', 'tools']
                           
        for kc in known_categories:
            categories.add(kc)
            
        categories = sorted(list(categories))
        
        # The user already requested "aviation" top 20, we can skip it or just do it again (since it might skip ones it already downloaded if we track them or if the user doesn't care).
        # We'll skip aviation for efficiency
        if 'aviation' in categories:
            categories.remove('aviation')
            
        print(f"Found {len(categories)} categories to process: {', '.join(categories)}", flush=True)
        
    except Exception as e:
        print(f"Failed to extract categories: {e}", flush=True)
        driver.quit()
        return

    visited_models = set()

    for category in categories:
        print(f"\n==========================================================", flush=True)
        print(f"Starting downloads for category: {category}", flush=True)
        print(f"==========================================================", flush=True)
        
        downloaded_count = 0
        page = 1
        
        # create category folder if not exists
        # Actually save everything into one folder, or separate folders? Let's put in subfolders!
        cat_dir = os.path.join(download_dir, category)
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)
            
        # Update driver prefs dynamically? Selenium doesn't let you update prefs without recreating the driver.
        # We'll just download to the main dir, and if the user wants they can sort them. Or we just leave them.
        
        while downloaded_count < 20 and page <= 5: # Limit to first 5 pages per category to avoid infinite loops
            print(f"Scraping {category} search page {page}...", flush=True)
            driver.get(f"https://grabcad.com/library?page={page}&time=this_month&sort=most_liked&categories={category}")
            time.sleep(4)
            
            links = driver.find_elements(By.CSS_SELECTOR, "a[href^='/library/']")
            hrefs = []
            for link in links:
                try:
                    h = link.get_attribute("href")
                    if h:
                        hrefs.append(h)
                except:
                    pass
                    
            unique_hrefs = []
            for h in hrefs:
                if h not in visited_models:
                    if "/users/" not in h and "/software/" not in h and "/categories/" not in h and "/challenges/" not in h:
                        unique_hrefs.append(h)
                        visited_models.add(h)
                        
            if not unique_hrefs:
                print("Could not find new model links on this page.", flush=True)
                break
                
            print(f"Found {len(unique_hrefs)} potential links.", flush=True)
                
            for href in unique_hrefs:
                if downloaded_count >= 20:
                    break
                    
                print(f"[{category} {downloaded_count+1}/20] Checking model: {href}", flush=True)
                try:
                    driver.get(href)
                    time.sleep(3)
                except Exception as e:
                    print(f"Failed to navigate: {e}", flush=True)
                    continue
                
                try:
                    body_text = driver.find_element(By.TAG_NAME, "body").text.lower()
                except:
                    print("Could not read body text.", flush=True)
                    continue
                    
                if ".step" in body_text or ".stp" in body_text or ".stl" in body_text or "step / iges" in body_text or "stl" in body_text:
                    print(" -> Formats match. Attempting download...", flush=True)
                    try:
                        download_clicked = False
                        
                        elements = driver.find_elements(By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download file')]")
                        for elem in elements:
                            if elem.is_displayed():
                                elem.click()
                                print(" -> Download initiated!", flush=True)
                                downloaded_count += 1
                                download_clicked = True
                                time.sleep(10)
                                break
                        
                        if not download_clicked:
                            links = driver.find_elements(By.TAG_NAME, "a")
                            for link in links:
                                if "download" in link.text.lower() and "file" in link.text.lower() and link.is_displayed():
                                    link.click()
                                    print(" -> Download initiated via fallback!", flush=True)
                                    downloaded_count += 1
                                    download_clicked = True
                                    time.sleep(10)
                                    break
                                    
                        if not download_clicked:
                            print(" -> Could not find download button.", flush=True)
                            
                    except Exception as e:
                        print(f" -> Error during download click: {e}", flush=True)
                else:
                    print(" -> Formats do not match.", flush=True)
                    
            page += 1

    print("\nAll categories processed!", flush=True)
    print("Waiting 60 seconds for pending downloads to finish...", flush=True)
    time.sleep(60)
    driver.quit()
    print("Browser closed.", flush=True)

if __name__ == "__main__":
    main()
