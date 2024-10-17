import random
import nltk
from nltk import word_tokenize
from collections import defaultdict

# Load and preprocess the text
def load_corpus(file_path):
    with open(file_path, 'r') as file:
        text = file.read().lower()
    return word_tokenize(text)

# Build the Markov chain model
def build_model(tokens):
    model = defaultdict(list)
    for i in range(len(tokens) - 1):
        model[tokens[i]].append(tokens[i + 1])
    return model

# Predict the next word
def predict_next_word(model, current_word):
    if current_word in model:
        return random.choice(model[current_word])
    else:
        return None

def main():
    nltk.download('punkt')  # Download NLTK tokenizer
    tokens = load_corpus('corpus.txt')
    model = build_model(tokens)

    while True:
        current_word = input("Enter a word (or 'exit' to quit): ").lower()
        if current_word == 'exit':
            break
        next_word = predict_next_word(model, current_word)
        if next_word:
            print(f"Next word prediction: {next_word}")
        else:
            print("No prediction available for that word.")

if __name__ == "__main__":
    main()
