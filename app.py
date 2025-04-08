from asyncio import Task
import uuid
import json
import time
import numpy as np
from threading import Thread
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from transformers import GPT2Tokenizer, PreTrainedTokenizerFast

# Your existing imports for scraping and sentiment analysis functions...
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

app = Flask(__name__)

# Load your models.
model1 = load_model("model1.h5")
model2 = load_model("model2.h5")
model3 = load_model("model3.h5")

# Global dictionary to store task statuses and results.
tasks = {}

#########################################
#           SCRAPER FUNCTION            #
#########################################
def scrape_reddit_comments_only(subreddit, max_posts, keyword, output_file):
    # Dummy sentiment analysis function for now.
    def sentiment_analysis(_):
        return "positive"

    if keyword == "none" or keyword == "null" or keyword == "None":
        keyword = "*"

    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 20)
    base_url = f'https://old.reddit.com/r/{subreddit}/search?q={keyword}&restrict_sr=on&sort=new'
    driver.get(base_url)

    processed_links = set()
    post_links = []
    all_comments_data = []

    try:
        while len(post_links) < max_posts:
            try:
                wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.search-result')))
                time.sleep(2)
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

        for link in post_links:
            try:
                driver.get(link)
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div#siteTable')))
                time.sleep(2)
                try:
                    original_post_element = driver.find_element(By.CSS_SELECTOR, 'div#siteTable div.thing:not(.comment)')
                    try:
                        original_post_text = original_post_element.find_element(By.CSS_SELECTOR, 'div.usertext-body div.md').text.strip()
                    except:
                        original_post_text = ""
                except:
                    original_post_text = ""
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
                                time_str = time_element.get_attribute('datetime')
                                day_posted = time_str.split('T')[0]
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
                except Exception as e:
                    print(f"No comments or error scraping comments for post: {link}", e)
            except Exception as e:
                print(f"Error loading post: {link}", e)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_comments_data, f, ensure_ascii=False, indent=4)
    finally:
        driver.quit()

#########################################
#       SENTIMENT PREDICTION CODE       #
#########################################
def predict_sentiments_on_comments(json_file, chosen_model):
    with open(json_file, 'r', encoding='utf-8') as f:
        comments_data = json.load(f)

    
   

    if chosen_model != model2:
        sequence_length = 200 
    else:
        sequence_length = 500

    if chosen_model != model3:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
        tokenizer.pad_token = "<pad>"
        tokenizer.unk_token = "<unk>"

    label_mapping_rev = {0: "negative", 1: "neutral", 2: "positive"}

    for comment in comments_data:
        comment_text = comment.get("comment", "")
        post_text = comment.get("original_post", "")
        comment_tokens = tokenizer.encode(comment_text, max_length=sequence_length, truncation=True, padding='max_length')
        post_tokens = tokenizer.encode(post_text, max_length=sequence_length, truncation=True, padding='max_length')
        combined_text = "POST: " + post_text + "\nCOMMENT: " + comment_text
        text_tokens = tokenizer.encode(combined_text, max_length=sequence_length, truncation=True, padding='max_length')

        if not comment_tokens:
            comment_tokens = [tokenizer.pad_token_id] * sequence_length
        if not post_tokens:
            post_tokens = [tokenizer.pad_token_id] * sequence_length
        if not text_tokens:
            text_tokens = [tokenizer.pad_token_id] * sequence_length

        comment_tokens = np.array(comment_tokens).reshape(1, sequence_length)
        post_tokens = np.array(post_tokens).reshape(1, sequence_length)
        text_tokens = np.array(text_tokens).reshape(1, sequence_length)

        if chosen_model != model2:
            pred = chosen_model.predict([comment_tokens, post_tokens])
        else:
            pred = chosen_model.predict(text_tokens)
        predicted_label = np.argmax(pred, axis=1)[0]
        comment["sentiment"] = label_mapping_rev[predicted_label]

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
        "negative": round(counts["negative"] * 100 / total, 2),
        "neutral": round(counts["neutral"] * 100 / total, 2),
        "positive": round(counts["positive"] * 100 / total, 2)
    }
    return percentages

#########################################
#         BACKGROUND TASK LOGIC         #
#########################################
def background_task(task_id, stock_name, max_posts, chosen_model):
    subreddit = "stocks"
    output_file = f"scraped_comments_{task_id}.json"
    scrape_reddit_comments_only(subreddit, max_posts, stock_name, output_file)
    comments_data = predict_sentiments_on_comments(output_file, chosen_model)
    percentages = calculate_percentages(comments_data)
    tasks[task_id] = {"status": "complete", "percentages": percentages, "total": len(comments_data)}

#########################################
#             FLASK ROUTES              #
#########################################
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        chosen_model_str = request.form.get('model_choice', 'model1')
        if chosen_model_str == 'model1':
            chosen_model = model1
        elif chosen_model_str == 'model2':
            chosen_model = model2
        else:
            chosen_model = model3

        stock_name = request.form.get('stock_name')
        try:
            max_posts = int(request.form.get('max_posts'))
        except ValueError:
            max_posts = 100

        # Create a unique task ID.
        task_id = str(uuid.uuid4())
        tasks[task_id] = {"status": "processing"}

        # Start the background thread.
        thread = Thread(target=background_task, args=(task_id, stock_name, max_posts, chosen_model))
        thread.start()

        # Redirect to the progress page.
        return redirect(url_for('progress', task_id=task_id))
    return render_template('index.html')

@app.route('/progress/<task_id>')
def progress(task_id):
    return render_template('progress.html', task_id=task_id)

@app.route('/status/<task_id>')
def status(task_id):
    # Returns the current status of the task in JSON.
    if task_id in tasks:
        return jsonify(tasks[task_id])
    else:
        return jsonify({"status": "unknown"})

@app.route('/result/<task_id>')
def result(task_id):
    # Only render results if the task is complete.
    if task_id in tasks and tasks[task_id].get("status") == "complete":
        # Remove the task entry from memory after use.
        result_data = tasks.get(task_id)
        return render_template('result.html',
                               percentages=result_data["percentages"],
                               total=result_data["total"])
    else:
        return "Task is not complete yet", 202

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
