import numpy as np
from collections import Counter
import random

def load_large_text_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def text_to_signal(text):
    words = text.split()
    unique_words = list(set(words))
    word_to_index = {word: idx for idx, word in enumerate(unique_words)}
    index_to_word = {idx: word for word, idx in word_to_index.items()}
    signal = np.array([word_to_index[word] for word in words])
    return signal, word_to_index, index_to_word

def get_mode_intervals(mode):
    modes = {
        'Ionian': [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8],
        'Dorian': [1, 9/8, 6/5, 4/3, 3/2, 5/3, 13/8],
        'Lydian': [1, 9/8, 5/4, 45/32, 3/2, 5/3, 15/8],
        'Mixolydian': [1, 9/8, 5/4, 4/3, 3/2, 27/16, 7/4],
        'Aeolian': [1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5],
    }
    return modes.get(mode, modes['Ionian'])

def build_ngram_model(text, n=3):
    words = text.split()
    ngrams = {}
    for i in range(len(words)-n+1):
        key = tuple(words[i:i+n-1])
        next_word = words[i+n-1]
        if key not in ngrams:
            ngrams[key] = []
        ngrams[key].append(next_word)
    return ngrams

class ResonantNodeText:
    def __init__(self, harmonic_ratio, node_id=0):
        self.base_ratio = harmonic_ratio
        self.node_id = node_id

    def compute(self, time_step):
        # Modulate the harmonic ratio over time
        self.harmonic_ratio = self.base_ratio * (1 + 0.1 * np.sin(2 * np.pi * 0.05 * time_step))
        frequency = self.harmonic_ratio
        value = np.sin(2 * np.pi * frequency * time_step)
        return value

def generate_text_from_ngrams(ngrams, start_words, num_words, nodes):
    text = list(start_words)
    current_words = tuple(start_words)
    time_step = 0
    time_increment = 1.0 / num_words

    for _ in range(num_words - len(start_words)):
        next_words = ngrams.get(current_words, None)
        if not next_words:
            break

        node_outputs = [node.compute(time_step) for node in nodes]
        probabilities = np.abs(node_outputs)
        # Ensure probabilities sum to 1 to prevent division by zero
        prob_sum = probabilities.sum()
        if prob_sum > 0:
            probabilities /= prob_sum
        else:
            probabilities = np.ones_like(probabilities) / len(probabilities)

        # Map next words to probabilities based on word lengths
        word_probs = []
        for word in next_words:
            word_length = len(word)
            idx = word_length % len(probabilities)
            prob = probabilities[idx]
            word_probs.append(prob)
        total = sum(word_probs)
        word_probs = [p / total for p in word_probs]

        next_word = random.choices(next_words, weights=word_probs, k=1)[0]
        text.append(next_word)
        current_words = tuple(text[-(len(current_words)):])
        time_step += time_increment
    return ' '.join(text)

def main():
    # Step 1: Load a large text corpus
    corpus_text = load_large_text_corpus('alice_in_wonderland.txt')
    
    # Step 2: Convert text to signal
    signal, word_to_index, index_to_word = text_to_signal(corpus_text)
    
    # Step 3: Build n-gram model
    ngram_model = build_ngram_model(corpus_text, n=3)
    
    # Step 4: Analyze the text
    counts = Counter(signal)
    most_common = counts.most_common()
    dominant_indices = [item[0] for item in most_common[:7]]  # Match number of harmonic ratios
    dominant_words = [index_to_word[idx] for idx in dominant_indices]
    
    for selected_mode in ["Ionian", "Dorian", "Lydian", "Mixolydian", "Aeolian"]:
        # Step 5: Initialize nodes with harmonic ratios
        harmonic_ratios = get_mode_intervals(selected_mode)
        nodes = [ResonantNodeText(harmonic_ratio=ratio, node_id=i+1) for i, ratio in enumerate(harmonic_ratios)]
        
        # Step 6: Generate new text
        start_words = random.sample(dominant_words, 2)
        generated_text = generate_text_from_ngrams(ngram_model, start_words, num_words=1000, nodes=nodes)
        print(f"Generated Text (Mode: {selected_mode}):\n{generated_text}")
    
if __name__ == "__main__":
    main()
