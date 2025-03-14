from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import json
import time

def scrape_reddit_comments_only(subreddit, max_posts, keyword, output_file):
   
    # Placeholder sentiment analysis function
    def sentiment_analysis(_):
        return "positive"

    if keyword is None:
        keyword = "*"

    # Initialize WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 20)

    # Open the search results page (sorted by "new")
    base_url = f'https://old.reddit.com/r/{subreddit}/search?q={keyword}&restrict_sr=on&sort=new'
    driver.get(base_url)

    processed_links = set()
    post_links = []  # Will store the list of post links we need to scrape
    all_comments_data = []

    try:
        # Step 1: Gather up to `max_posts` post links
        while len(post_links) < max_posts:
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.search-result')))
                time.sleep(2)  # Let content stabilize
            except:
                print("Error: Posts did not load properly.")
                break

            posts = driver.find_elements(By.CSS_SELECTOR, 'div.search-result')
            for post in posts:
                try:
                    title_element = post.find_element(By.CSS_SELECTOR, 'a.search-title')
                    post_link = title_element.get_attribute('href')
                    post_title = title_element.text

                    post_identifier = (post_title, post_link)
                    if post_identifier not in processed_links:
                        processed_links.add(post_identifier)
                        post_links.append(post_link)

                    if len(post_links) >= max_posts:
                        break
                except:
                    continue

            if len(post_links) >= max_posts:
                break

            try:
                next_button = driver.find_element(By.CSS_SELECTOR, 'a[rel="nofollow next"]')
                driver.execute_script("arguments[0].scrollIntoView();", next_button)
                time.sleep(2)
                next_button.click()
                time.sleep(3)
            except:
                print("No more pages available or error clicking the next button.")
                break

        # Step 2: For each post, scrape its original post text and its comments
        for link in post_links:
            try:
                driver.get(link)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div#siteTable')))
                time.sleep(2)

                # Extract the original post text
                
                try:
                    original_post_element = driver.find_element(By.CSS_SELECTOR, 'div#siteTable div.thing:not(.comment)')
                    try:
                        original_post_text = original_post_element.find_element(By.CSS_SELECTOR, 'div.usertext-body div.md').text.strip()
                    except:
                        original_post_text = ""
                except:
                    original_post_text = ""

                # Scrape comments
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.commentarea')))
                    comment_elements = driver.find_elements(By.CSS_SELECTOR, 'div.comment')

                    for c in comment_elements:
                        # Get the username
                        try:
                            author_element = c.find_element(By.CSS_SELECTOR, 'a.author')
                            username = author_element.text
                        except:
                            username = "[deleted]"

                        # Get the comment text
                        try:
                            body_element = c.find_element(By.CSS_SELECTOR, 'div.usertext-body div.md')
                            comment_text = body_element.text.strip()
                        except:
                            comment_text = ""

                        if comment_text:
                            # Extract the posting date
                            try:
                                time_element = c.find_element(By.CSS_SELECTOR, 'time')
                                time_str = time_element.get_attribute('datetime')  # e.g. "2024-01-01T12:34:56+00:00"
                                day_posted = time_str.split('T')[0]  # keep only YYYY-MM-DD
                            except:
                                day_posted = "unknown"

                            sentiment = sentiment_analysis(comment_text)

                            all_comments_data.append({
                                "username": username,
                                "comment": comment_text,
                                "day_posted": day_posted,
                                "sentiment": sentiment,
                                "real_value": "",
                                "original_post": original_post_text
                            })
                except:
                    print(f"No comments or error scraping comments for post: {link}")
            except:
                print(f"Error loading post: {link}")

        # Step 3: Write comments to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_comments_data, f, ensure_ascii=False, indent=4)

    finally:
        driver.quit()

# Example usage
if __name__ == "__main__":
    scrape_reddit_comments_only(
        subreddit='stocks',
        max_posts=1000,
        keyword=None,
        output_file='new_not_labeled.json'
    )
