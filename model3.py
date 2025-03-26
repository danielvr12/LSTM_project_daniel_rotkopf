import json
import random
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
# Import Hugging Face tokenizers for BPE.
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Hyperparameters
sequence_length = 200        # Maximum sequence length for tokens
embedding_dim = 128          # Embedding dimension
batch_size = 64              # Batch size for training
epochs = 100                  # Number of training epochs

# 1. Load and parse your JSON dataset.
with open("labeled_comments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# The JSON file is expected to have a top-level key "comments"
comments = data["comments"]

# 2. Prepare texts for training the BPE tokenizer (both comments and original posts).
texts = [entry["comment"] for entry in comments] + [entry["original_post"] for entry in comments]

# 3. Train a BPE tokenizer on your dataset.
# Initialize a BPE tokenizer with an unknown token.
bpe_tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
# Use a byte-level pre-tokenizer
bpe_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# Set up a trainer with a target vocabulary size of 50k and special tokens.
trainer = trainers.BpeTrainer(vocab_size=100000, special_tokens=["<pad>", "<unk>"])
bpe_tokenizer.train_from_iterator(texts, trainer=trainer)

# Define our pad token and get its ID.
pad_token = "<pad>"
unk_token = "<unk>"
pad_token_id = bpe_tokenizer.token_to_id(pad_token)
bpe_tokenizer.save("tokenizer.json")

# 4. Extract (comment, original_post) pairs and map labels to integers.
text_pairs = [(entry["comment"], entry["original_post"]) for entry in comments]
label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
labels = [label_mapping[entry["label"].lower()] for entry in comments]

# 5. Balance the dataset (undersample overrepresented classes).
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

# 6. Create a tf.data.Dataset from the balanced data.
dataset = tf.data.Dataset.from_tensor_slices((balanced_text_pairs, balanced_labels))
dataset = dataset.shuffle(buffer_size=len(balanced_text_pairs), seed=42)

# 7. Define tokenization functions using the BPE tokenizer.
def tokenize_text(text):
    def encode_fn(t):
        # Convert the tensor to a numpy value.
        t_val = t.numpy()
        # Decode bytes to string if needed.
        if isinstance(t_val, bytes):
            t_str = t_val.decode('utf-8')
        else:
            t_str = str(t_val)
        # If the string is empty, return a sequence of pad tokens.
        if not t_str.strip():
            encoded = [pad_token_id] * sequence_length
        else:
            # Encode the text using the BPE tokenizer.
            encoding = bpe_tokenizer.encode(t_str)
            encoded = encoding.ids
            # Pad or truncate to the fixed sequence length.
            if len(encoded) < sequence_length:
                encoded = encoded + [pad_token_id] * (sequence_length - len(encoded))
            else:
                encoded = encoded[:sequence_length]
        return np.array(encoded, dtype=np.int32)
    
    tokens = tf.py_function(func=encode_fn, inp=[text], Tout=tf.int32)
    tokens.set_shape([sequence_length])
    return tokens

def tokenize_pair(text_pair, label):
    # text_pair is a tuple: (comment, original_post)
    comment_tokens = tokenize_text(text_pair[0])
    post_tokens = tokenize_text(text_pair[1])
    return (comment_tokens, post_tokens), label

# Apply tokenization to the dataset.
tokenized_dataset = dataset.map(tokenize_pair)

# 8. Split the dataset into training and validation sets (80% train, 20% validation).
total_samples = len(balanced_text_pairs)
train_size = int(0.8 * total_samples)
train_dataset = tokenized_dataset.take(train_size)
val_dataset = tokenized_dataset.skip(train_size)

# 9. Build the model.
# Two separate inputs: one for comment tokens and one for original post tokens.
vocab_size = bpe_tokenizer.get_vocab_size()
input_comment = layers.Input(shape=(sequence_length,), dtype=tf.int32, name="comment_input")
input_post = layers.Input(shape=(sequence_length,), dtype=tf.int32, name="post_input")

# Each branch uses its own embedding layer.
comment_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                     name="comment_embedding")(input_comment)
post_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                  name="post_embedding")(input_post)

# Mask the padded tokens.
comment_masked = layers.Masking(mask_value=pad_token_id)(comment_embedding)
post_masked = layers.Masking(mask_value=pad_token_id)(post_embedding)

# Process each sequence with its own LSTM.
comment_lstm = layers.LSTM(64, name="comment_lstm")(comment_masked)
post_lstm = layers.LSTM(64, name="post_lstm")(post_masked)

# Concatenate the outputs and add a classification layer.
concatenated = layers.concatenate([comment_lstm, post_lstm])
outputs = layers.Dense(3, activation="softmax")(concatenated)

model = tf.keras.models.Model(inputs=[input_comment, input_post], outputs=outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# 10. Batch and prefetch the datasets.
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 11. Train the model.
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# 12. Plot training & validation accuracy.
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# 13. Plot training & validation loss.
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Save the trained model.
model.save("model33.h5", save_format='h5')
