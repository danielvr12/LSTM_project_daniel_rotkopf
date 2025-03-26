import json
import random
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Initialize GPT-2 tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT-2 does not have a pad token by default; set it to the EOS token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Hyperparameters
sequence_length = 400     # Maximum sequence length for tokens
embedding_dim = 256         # Dimension for embedding layers
batch_size = 64              # Batch size for training
epochs = 50                 # Number of training epochs
vocab_size = tokenizer.vocab_size  # GPT-2 vocabulary size (~50257)

# 1. Load and parse your JSON dataset.
with open("labeled_comments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# The JSON file is expected to have a top-level key "comments"
comments = data["comments"]

# 2. Extract (comment, original_post) pairs and map labels to integers.
text_pairs = [(entry["comment"], entry["original_post"]) for entry in comments]
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
labels = [label_mapping[entry["label"].lower()] for entry in comments]

# 3. Balance the dataset (undersample overrepresented classes).
data_by_label = defaultdict(list)
for pair, label in zip(text_pairs, labels):
    data_by_label[label].append(pair)
min_count = min(len(text_list) for text_list in data_by_label.values())
print("Minimum count per class:", min_count)
balanced_text_pairs = []
balanced_labels = []
for label, text_list in data_by_label.items():
    sampled = random.sample(text_list, min_count) if len(text_list) > min_count else text_list
    balanced_text_pairs.extend(sampled)
    balanced_labels.extend([label] * len(sampled))
combined = list(zip(balanced_text_pairs, balanced_labels))
random.shuffle(combined)
balanced_text_pairs, balanced_labels = zip(*combined)
balanced_text_pairs = list(balanced_text_pairs)
balanced_labels = list(balanced_labels)

# 4. Create a tf.data.Dataset from the balanced data.
dataset = tf.data.Dataset.from_tensor_slices((balanced_text_pairs, balanced_labels))
dataset = dataset.shuffle(buffer_size=len(balanced_text_pairs), seed=42)

# 5. Define tokenization functions using the GPT-2 tokenizer.
def tokenize_text(text):
    def encode_fn(t):
        # Convert the EagerTensor to a numpy value.
        t_val = t.numpy()
        # Decode if it's a byte string.
        if isinstance(t_val, bytes):
            t_str = t_val.decode('utf-8')
        else:
            t_str = str(t_val)
        # If the string is empty or whitespace, return a padded sequence.
        if not t_str.strip():
            encoded = [tokenizer.pad_token_id] * sequence_length
        else:
            encoded = tokenizer.encode(
                t_str,
                max_length=sequence_length,
                truncation=True,
                padding='max_length'
            )
        # Safety check: if encoded somehow is empty, pad it.
        if len(encoded) == 0:
            encoded = [tokenizer.pad_token_id] * sequence_length
        return np.array(encoded, dtype=np.int32)
    
    tokens = tf.py_function(func=encode_fn, inp=[text], Tout=tf.int32)
    tokens.set_shape([sequence_length])
    return tokens

def tokenize_pair(text_pair, label):
    # text_pair is a tuple: (comment, original_post)
    post_text = text_pair[1]
    comment_text = text_pair[0]
    combined_text = "POST: " + post_text + "\nCOMMENT: " + comment_text
    tokens = tokenize_text(combined_text)
    return tokens, label

# Map tokenization to each sample.
tokenized_dataset = dataset.map(tokenize_pair)

# 6. Split the dataset into training and validation sets (80% train, 20% validation).
total_samples = len(balanced_text_pairs)
train_size = int(0.8 * total_samples)
train_dataset = tokenized_dataset.take(train_size)
val_dataset = tokenized_dataset.skip(train_size)

# 7. Build the model.
# Two separate inputs: one for the comment tokens and one for the original post tokens.
input_text = layers.Input(shape=(sequence_length,), dtype=tf.int32, name="combined_input")


# Each branch uses its own embedding layer (separate learnable parameters).
embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                     name="embedding")(input_text)


# Optionally, add a masking layer if you wish to mask padded tokens.
masking = layers.Masking(mask_value=tokenizer.pad_token_id)(embedding)


# Process each sequence with its own LSTM.
lstm = layers.LSTM(256, name="comment_lstm")(masking)

dense1 = layers.Dense(256, activation='relu', name="dense1")(lstm)
# Concatenate the outputs from both branches.
dense2 = layers.Dense(128, activation='relu', name="dense2")(dense1)
outputs = layers.Dense(3, activation="softmax")(dense2)

model = models.Model(inputs=input_text, outputs=outputs)

model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
model.summary()

# 8. Batch and prefetch the datasets.
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 9. Train the model.
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)


# 10. Plot training & validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# 11. Plot training & validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()



#save the model.
model.save("model2.h5", save_format='h5')
