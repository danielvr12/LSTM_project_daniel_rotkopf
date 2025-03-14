import openai
import json
import sqlite3
import os
import sys
import time

# Initialize the OpenAI API key
openai.api_key = "my_open_ai_key"  # didn't include my key for security reasons

PROGRESS_FILE = "progress.json"
LABELED_FILE = "labeled_comments.json"


def classify_comments(comment_objs):
  
    labels = []
    for comment_obj in comment_objs:
        comment_text = comment_obj.get('comment', '')
        original_post = comment_obj.get('original_post', '')
        # Prepare a prompt that includes both the original post context and the comment.
        prompt_text = (
            "You are a helpful assistant. Below, you are given two pieces of text talking about stocks:\n\n"
            "1. An Original Post, which provides the context.\n"
            "2. A Comment, which is a response to the original post.\n\n"
            "Please analyze both texts together and classify the overall commenter's sentiment about the stock as "
            "positive, negative, or neutral. Your response must be exactly one word (either 'positive', 'negative', or 'neutral').\n\n"
            f"Original Post: {original_post}\n\n"
            f"Comment: {comment_text}\n\n"
            "Respond with only one word."
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies overall sentiment based on provided context."},
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=10,
                temperature=0.0
            )
            label = response['choices'][0]['message']['content'].strip().lower()
        except Exception as e:
            # If an error occurs, re-raise it so it can be handled in main()
            raise e

        # Validate the label and default to neutral if unexpected
        if label not in ["positive", "negative", "neutral"]:
            label = "neutral"
        labels.append(label)
    return labels

def load_progress():
    """
    Loads progress from a JSON file if it exists. 
    Returns the index of the last processed batch and the partial labeled data if any.
    """
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
            return progress_data.get("last_batch_index", 0), progress_data.get("labeled_comments", [])
    else:
        return 0, []

def save_progress(last_batch_index, labeled_comments):
    
    progress_data = {
        "last_batch_index": last_batch_index,
        "labeled_comments": labeled_comments
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=4)

def main():
    # Load JSON data with utf-8 encoding
    with open('top_not_labeled.json', 'r', encoding='utf-8') as f:
        comments_data = json.load(f)

    # Check that comments_data is a list
    if not isinstance(comments_data, list):
        print("Error: comments_data is not a list. Please check the JSON file structure.")
        sys.exit(1)

    
    

    # Prepare comments in batches of 10
    batch_size = 10
    batches = [comments_data[i: i + batch_size] for i in range(0, len(comments_data), batch_size)]

    # Load progress (to resume if necessary)
    last_batch_index, labeled_comments = load_progress()

    total_batches = len(batches)
    print(f"Total Batches: {total_batches}, Resuming from batch {last_batch_index}.")

    try:
        for batch_index in range(last_batch_index, total_batches):
            batch = batches[batch_index]
            # Classify the overall sentiment for this batch (using both the original post and the comment)
            labels = classify_comments(batch)
            # Add the new label to each comment object and update our list of labeled comments
            for idx, label in enumerate(labels):
                # Remove the existing "sentiment" key if it exists
                if "sentiment" in batch[idx]:
                    del batch[idx]["sentiment"]
                batch[idx]['label'] = label
                labeled_comments.append(batch[idx])
                print(f"Processed comment #{len(labeled_comments)}")
                
            # Save progress (batch-by-batch)
            save_progress(batch_index + 1, labeled_comments)
            time.sleep(2)

        # Once all batches are processed, write the labeled data to a final JSON
        with open(LABELED_FILE, 'w', encoding='utf-8') as outfile:
            json.dump({"comments": labeled_comments}, outfile, indent=4, ensure_ascii=False)

        print(f"All comments labeled and saved to {LABELED_FILE}.")

    except Exception as e:
        # If an exception occurs, save the current progress and exit
        print("An error occurred. Saving progress before exiting.")
        save_progress(batch_index, labeled_comments)
        print(f"Saved progress up to batch {batch_index}. Error was: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
