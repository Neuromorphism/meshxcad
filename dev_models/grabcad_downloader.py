import sys
import time
import os
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

    driver.get("https://grabcad.com/library?page=1&time=this_month&sort=most_liked&categories=aviation")

    # Prompt on stdin, using flush to ensure it prints
    print("==========================================================", flush=True)
    print("A browser window has been opened.", flush=True)
    print("Please log in to GrabCAD manually in that window.", flush=True)
    print("Press Enter in THIS terminal when you are fully logged in...", flush=True)
    print("==========================================================", flush=True)
    sys.stdin.readline()

    downloaded_count = 0
    page = 1
    visited_models = set()

    while downloaded_count < 20:
        print(f"\nScraping search page {page}...", flush=True)
        driver.get(f"https://grabcad.com/library?page={page}&time=this_month&sort=most_liked&categories=aviation")
        time.sleep(4)
        
        # Try finding model links
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
                # Exclude non-model pages
                if "/users/" not in h and "/software/" not in h and "/categories/" not in h and "/challenges/" not in h:
                    # Also exclude things ending in -1 if they're not models, though grabcad links are usually model names
                    unique_hrefs.append(h)
                    visited_models.add(h)
                    
        # GrabCAD does have a 'model-name' div or similar, but unique URLs is usually enough
        
        if not unique_hrefs:
            print("Could not find new model links on this page or selector needs adjustment.", flush=True)
            break
            
        print(f"Found {len(unique_hrefs)} potential links on page {page}.", flush=True)
            
        for href in unique_hrefs:
            if downloaded_count >= 20:
                break
                
            print(f"Checking model: {href}", flush=True)
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
                    # First check for the standard big Download button
                    download_clicked = False
                    
                    # Usually it's a link or button containing "Download"
                    # Let's search by xpath text for robust matching
                    elements = driver.find_elements(By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'download file')]")
                    for elem in elements:
                        if elem.is_displayed():
                            elem.click()
                            print(" -> Download initiated!", flush=True)
                            downloaded_count += 1
                            download_clicked = True
                            time.sleep(10) # Wait for zip generation / download to start
                            break
                    
                    if not download_clicked:
                        # Fallback simple search
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
                print(" -> Formats do not match (No step/stl found on page).", flush=True)
                
        page += 1

    print(f"\nCompleted! Successfully initiated {downloaded_count} downloads.", flush=True)
    print("Waiting 30 seconds for pending downloads to finish...", flush=True)
    time.sleep(30)
    driver.quit()
    print("Browser closed.", flush=True)

if __name__ == "__main__":
    main()
