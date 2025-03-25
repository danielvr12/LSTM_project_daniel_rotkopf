from tkinter import ROUND
from flask import Flask, render_template, request
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from transformers import GPT2Tokenizer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

app = Flask(__name__)

# Global settings and model/tokenizer initialization
sequence_length = 200  # Must match the value used during training

# Initialize GPT-2 tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load your trained sentiment analysis model.
# Ensure that the directory "sentiment_lstm_model" exists and contains your saved model.
model1 = models.load_model("sentiment_lstm_model2",compile=False)
model2 = models.load_model("model2",compile=False)
model3 = models.load_model("model3",compile=False)


#########################################
#           SCRAPER FUNCTION            #
#########################################
def scrape_reddit_comments_only(subreddit, max_posts, keyword, output_file):
    """
    Scrapes comments from a subreddit (via old.reddit.com search) using the given keyword.
    Saves a JSON file with comment data (including a placeholder sentiment that will be overwritten).
    """
    # A dummy sentiment analysis function (will be replaced later)
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
    post_links = []  # List of post URLs to scrape
    all_comments_data = []

    try:
        # Step 1: Gather up to `max_posts` post links.
        while len(post_links) < max_posts:
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.search-result')))
                time.sleep(2)  # Let the content stabilize
            except Exception as e:
                print("Error: Posts did not load properly.", e)
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
                except Exception as e:
                    continue

            if len(post_links) >= max_posts:
                break

            try:
                next_button = driver.find_element(By.CSS_SELECTOR, 'a[rel="nofollow next"]')
                driver.execute_script("arguments[0].scrollIntoView();", next_button)
                time.sleep(2)
                next_button.click()
                time.sleep(3)
            except Exception as e:
                print("No more pages available or error clicking the next button.", e)
                break

        # Step 2: For each post, scrape the original post text and its comments.
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
                        try:
                            author_element = c.find_element(By.CSS_SELECTOR, 'a.author')
                            username = author_element.text
                        except:
                            username = "[deleted]"

                        try:
                            body_element = c.find_element(By.CSS_SELECTOR, 'div.usertext-body div.md')
                            comment_text = body_element.text.strip()
                        except:
                            comment_text = ""

                        if comment_text:
                            try:
                                time_element = c.find_element(By.CSS_SELECTOR, 'time')
                                time_str = time_element.get_attribute('datetime')  # Format
                                day_posted = time_str.split('T')[0]  # Keep only the date (YYYY-MM-DD)
                            except:
                                day_posted = "unknown"

                            sentiment = sentiment_analysis(comment_text)
                            all_comments_data.append({
                                "username": username,
                                "comment": comment_text,
                                "day_posted": day_posted,
                                "sentiment": sentiment,  # Placeholder, will be updated later
                                "real_value": "",
                                "original_post": original_post_text
                            })
                except Exception as e:
                    print(f"No comments or error scraping comments for post: {link}", e)
            except Exception as e:
                print(f"Error loading post: {link}", e)

        # Step 3: Write comments to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_comments_data, f, ensure_ascii=False, indent=4)
    finally:
        driver.quit()


#########################################
#       SENTIMENT PREDICTION CODE       #
#########################################
def predict_sentiments_on_comments(json_file,chosen_model):

    with open(json_file, 'r', encoding='utf-8') as f:
        comments_data = json.load(f)

    # Reverse mapping to convert prediction integer to sentiment label.
    label_mapping_rev = {0: "negative", 1: "neutral", 2: "positive"}

    for comment in comments_data:
        comment_text = comment.get("comment", "")
        post_text = comment.get("original_post", "")

        # Tokenize the comment and the original post using the same parameters as in training.
        comment_tokens = tokenizer.encode(comment_text, max_length=sequence_length, truncation=True, padding='max_length')
        post_tokens = tokenizer.encode(post_text, max_length=sequence_length, truncation=True, padding='max_length')

        #If tokenization returns an empty list, pad it.
        if not comment_tokens:
            comment_tokens = [tokenizer.pad_token_id] * sequence_length
        if not post_tokens:
            post_tokens = [tokenizer.pad_token_id] * sequence_length

        # Prepare batch dimension for model prediction.
        comment_tokens = np.array(comment_tokens).reshape(1, sequence_length)
        post_tokens = np.array(post_tokens).reshape(1, sequence_length)

        # Get model prediction.
        pred = chosen_model.predict([comment_tokens, post_tokens])
        predicted_label = np.argmax(pred, axis=1)[0]
        comment["sentiment"] = label_mapping_rev[predicted_label]

    #save the updated file.
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(comments_data, f, ensure_ascii=False, indent=4)

    return comments_data


def calculate_percentages(comments_data):
    
    counts = {"negative": 0, "neutral": 0, "positive": 0}
    total = len(comments_data)
    if total == 0:
        return counts
    for comment in comments_data:
        label = comment.get("sentiment", "neutral")
        counts[label] += 1
    percentages = {
        "negative":round(counts["negative"] * 100 / total,2),
        "neutral":round(counts["neutral"] * 100 / total,2),
        "positive":round(counts["positive"] * 100 / total,2)
        }
    return percentages


#########################################
#             FLASK ROUTES              #
#########################################
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. Capture which model was selected
        chosen_model_str = request.form.get('model_choice', 'model1')
        if chosen_model_str == 'model1':
            chosen_model = model1
        elif chosen_model_str == 'model2':
            chosen_model = model2
        else:
            chosen_model = model3

        # 2. Get other form inputs
        stock_name = request.form.get('stock_name')
        try:
            max_posts = int(request.form.get('max_posts'))
        except ValueError:
            max_posts = 100  # default if conversion fails

        # 3. Scrape
        subreddit = "stocks"
        output_file = "scraped_comments.json"
        scrape_reddit_comments_only(subreddit, max_posts, stock_name, output_file)

        # 4. Run sentiment prediction with the chosen model
        comments_data = predict_sentiments_on_comments(output_file, chosen_model)

        # 5. Calculate sentiment percentages
        percentages = calculate_percentages(comments_data)

        return render_template('result.html', percentages=percentages, total=len(comments_data))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)


