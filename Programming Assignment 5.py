#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Aidan Goodfellow
# Programming Assignment 5


# In[2]:


import re

def extract_story(filename, start_phrase, next_story_title):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

            # Find the first occurrence of the start phrase
            first_start_index = content.find(start_phrase)
            if first_start_index == -1:
                return "Start phrase not found."

            # Find the second occurrence of the start phrase
            second_start_index = content.find(start_phrase, first_start_index + 1)
            if second_start_index == -1:
                return "Second occurrence of start phrase not found."

            # Find the end phrase after the second occurrence of the start phrase
            end_index = content.find(next_story_title, second_start_index)
            if end_index == -1:
                return "End phrase not found."

            return content[second_start_index:end_index]
    except FileNotFoundError:
        return "File not found."

def process_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)

    # Split text into words
    words = text.split()

    return words

# Replace 'your_text_file.txt' with the path to your document
extracted_story = extract_story('C:\\Users\\aidan\\FA_23\\Applied Algorithms\\brothers_Grimm.txt', "THE FROG-PRINCE", "CAT AND MOUSE IN PARTNERSHIP")
word_list = process_text(extracted_story)

# Convert list of words to a set to remove duplicates
word_set = set(word_list)

print(word_set)
    


# In[4]:


def polynomial_rolling_hash(word, p=31, m=1024):
    hash_value = 0
    p_pow = 1
    for char in word:
        hash_value = (hash_value + (ord(char) - ord('a') + 1) * p_pow) % m
        p_pow = (p_pow * p) % m
    return hash_value

# Example usage
polynomial_hashes = {word: polynomial_rolling_hash(word) for word in word_set}
print(polynomial_hashes)


# In[3]:


class ChainingHashSet:
    def __init__(self, size=1024):
        self.size = size
        self.table = [[] for _ in range(size)]  # Create a list of empty lists

    def _hash(self, word, p=31, m=1024):
        # Polynomial rolling hash function
        hash_value = 0
        p_pow = 1
        for char in word:
            # Adjust the formula to handle full ASCII range
            hash_value = (hash_value + (ord(char) - ord('a') + 1) * p_pow) % m
            p_pow = (p_pow * p) % m
        return hash_value

    def add(self, word):
        hash_index = self._hash(word)
        if word not in self.table[hash_index]:
            self.table[hash_index].append(word)
            return True
        return False  # Word already exists

    def contains(self, word):
        hash_index = self._hash(word)
        return word in self.table[hash_index]

    def __len__(self):
        return sum(len(slot) for slot in self.table)

# Example usage
chaining_hash_set = ChainingHashSet()
for word in word_set:
    chaining_hash_set.add(word)

# Test if a word is in the hash set
print(chaining_hash_set.contains("frog"))  # Example word

# Get the number of words in the hash set
print(len(chaining_hash_set))


# In[9]:


class LinearProbingHashSet:
    def __init__(self, size=1024):
        self.size = size
        self.table = [None] * size  # Initialize the table with None
        self.total_insert_attempts = 0

    def _hash(self, word, p=31, m=1024):
        # Polynomial rolling hash function
        hash_value = 0
        p_pow = 1
        for char in word:
            hash_value = (hash_value + (ord(char) - ord('a') + 1) * p_pow) % m
            p_pow = (p_pow * p) % m
        return hash_value

    def _probe(self, hash_index):
        # Linear probing
        index = hash_index
        while self.table[index] is not None and index < self.size:
            index = (index + 1) % self.size
            if index == hash_index:
                return -1  # Indicates that the table is full
        return index

    def add(self, word):
        i = 0
        index = self._hash(word)
        while self.table[index] is not None and self.table[index] != word:
            i += 1
            index = (index + 1) % self.size
            if i == self.size:
                return False
        self.total_insert_attempts += i + 1  # Increment by the number of slots visited
        self.table[index] = word
        return True

    def contains(self, word):
        hash_index = self._hash(word)
        index = hash_index
        while self.table[index] is not None:
            if self.table[index] == word:
                return True
            index = (index + 1) % self.size
            if index == hash_index:
                break
        return False

    def __len__(self):
        return sum(1 for slot in self.table if slot is not None)

# Example usage
linear_probing_hash_set = LinearProbingHashSet()
for word in word_set:
    linear_probing_hash_set.add(word)

# Test if a word is in the hash set
print(linear_probing_hash_set.contains("frog"))  # Example word

# Get the number of words in the hash set
print(len(linear_probing_hash_set))


# In[10]:


class QuadraticProbingHashSet:
    def __init__(self, size=1024, c1=0.5, c2=0.5):
        self.size = size
        self.table = [None] * size  # Initialize the table with None
        self.c1 = c1
        self.c2 = c2
        self.total_insert_attempts = 0  # New attribute for tracking

    def _hash(self, word, p=31, m=1024):
        # Polynomial rolling hash function
        hash_value = 0
        p_pow = 1
        for char in word:
            hash_value = (hash_value + (ord(char) - ord('a') + 1) * p_pow) % m
            p_pow = (p_pow * p) % m
        return hash_value

    def _probe(self, hash_index, i):
        # Quadratic probing
        return (hash_index + int(self.c1 * i) + int(self.c2 * i * i)) % self.size

    def add(self, word):
        i = 0
        index = self._hash(word)
        while self.table[index] is not None and self.table[index] != word:
            i += 1
            index = self._probe(index, i)
            if i == self.size:
                return False
        self.total_insert_attempts += i + 1  # Increment by the number of slots visited
        self.table[index] = word
        return True

    def contains(self, word):
        i = 0
        index = self._hash(word)
        while self.table[index] is not None:
            if self.table[index] == word:
                return True
            i += 1
            index = self._probe(index, i)
            if i == self.size:
                break
        return False

    def __len__(self):
        return sum(1 for slot in self.table if slot is not None)

# Example usage
quadratic_probing_hash_set_half = QuadraticProbingHashSet(c1=0.5, c2=0.5)
quadratic_probing_hash_set_one = QuadraticProbingHashSet(c1=0, c2=1)

for word in word_set:
    quadratic_probing_hash_set_half.add(word)  # c1 = c2 = 0.5
    quadratic_probing_hash_set_one.add(word)   # c1 = 0; c2 = 1

# Test if a word is in the hash sets
print(quadratic_probing_hash_set_half.contains("frog"))  # c1 = c2 = 0.5
print(quadratic_probing_hash_set_one.contains("frog"))   # c1 = 0; c2 = 1

# Get the number of words in the hash sets
print(len(quadratic_probing_hash_set_half))  # c1 = c2 = 0.5
print(len(quadratic_probing_hash_set_one))   # c1 = 0; c2 = 1


# In[11]:


# Questions

#1. I was able to insert all words.

#2. 
max_chain_length = 0
unused_slots = 0

for slot in chaining_hash_set.table:
    if not slot:  # Check if the slot is empty
        unused_slots += 1
    else:
        chain_length = len(slot)
        if chain_length > max_chain_length:
            max_chain_length = chain_length

print("Maximum Chain Length:", max_chain_length)
print("Number of Unused Slots:", unused_slots)


# In[12]:


# For Linear Probing
average_nodes_visited_linear = linear_probing_hash_set.total_insert_attempts / len(linear_probing_hash_set)

# For Quadratic Probing
average_nodes_visited_quadratic_half = quadratic_probing_hash_set_half.total_insert_attempts / len(quadratic_probing_hash_set_half)
average_nodes_visited_quadratic_one = quadratic_probing_hash_set_one.total_insert_attempts / len(quadratic_probing_hash_set_one)

print("Average Nodes Visited (Linear Probing):", average_nodes_visited_linear)
print("Average Nodes Visited (Quadratic Probing c1=0.5, c2=0.5):", average_nodes_visited_quadratic_half)
print("Average Nodes Visited (Quadratic Probing c1=0, c2=1):", average_nodes_visited_quadratic_one)


# In[13]:




num_words_linear = len(linear_probing_hash_set)
num_words_quadratic_half = len(quadratic_probing_hash_set_half)
num_words_quadratic_one = len(quadratic_probing_hash_set_one)

print("Number of words inserted using Linear Probing:", num_words_linear)
print("Number of words inserted using Quadratic Probing (c1 = 0.5, c2 = 0.5):", num_words_quadratic_half)
print("Number of words inserted using Quadratic Probing (c1 = 0, c2 = 1):", num_words_quadratic_one)


# In[14]:


extracted_story = extract_story('C:\\Users\\aidan\\FA_23\\Applied Algorithms\\brothers_Grimm.txt', "CAT AND MOUSE IN PARTNERSHIP", "END OF THE PROJECT GUTENBERG EBOOK GRIMMS' FAIRY TALES")
word_list = process_text(extracted_story)

# Convert list of words to a set to remove duplicates
word_set = set(word_list)

print(word_set)


# In[15]:


for word in word_set:
    chaining_hash_set.add(word)


# In[16]:


for word in word_set:
    linear_probing_hash_set.add(word)


# In[17]:


for word in word_set:
    quadratic_probing_hash_set_half.add(word)  # c1 = c2 = 0.5
    quadratic_probing_hash_set_one.add(word)   # c1 = 0; c2 = 1


# In[18]:


num_elements_chaining = len(chaining_hash_set)
num_elements_linear = len(linear_probing_hash_set)
num_elements_quadratic_half = len(quadratic_probing_hash_set_half)
num_elements_quadratic_one = len(quadratic_probing_hash_set_one)

print("Number of elements inserted using Chaining:", num_elements_chaining)
print("Number of elements inserted using Linear Probing:", num_elements_linear)
print("Number of elements inserted using Quadratic Probing (c1 = 0.5, c2 = 0.5):", num_elements_quadratic_half)
print("Number of elements inserted using Quadratic Probing (c1 = 0, c2 = 1):", num_elements_quadratic_one)


# In[19]:


max_chain_length = 0
min_chain_length = float('inf')  # Set to infinity initially
max_chain_words = []
min_chain_words = []

for chain in chaining_hash_set.table:
    chain_length = len(chain)

    if chain_length > 0:  # Only consider non-empty chains
        if chain_length > max_chain_length:
            max_chain_length = chain_length
            max_chain_words = chain  # Store the words in the longest chain

        if chain_length < min_chain_length:
            min_chain_length = chain_length
            min_chain_words = chain  # Store the words in the shortest chain

# Assuming non-empty chains exist, pick a representative word from each
max_chain_example_word = max_chain_words[0] if max_chain_words else "None"
min_chain_example_word = min_chain_words[0] if min_chain_words else "None"

print("Maximum Chain Length:", max_chain_length, "Example Word:", max_chain_example_word)
print("Minimum Chain Length:", min_chain_length, "Example Word:", min_chain_example_word)


# In[20]:


# For Linear Probing
average_nodes_visited_linear = linear_probing_hash_set.total_insert_attempts / len(linear_probing_hash_set)

# For Quadratic Probing
average_nodes_visited_quadratic_half = quadratic_probing_hash_set_half.total_insert_attempts / len(quadratic_probing_hash_set_half)
average_nodes_visited_quadratic_one = quadratic_probing_hash_set_one.total_insert_attempts / len(quadratic_probing_hash_set_one)

print("Average Nodes Visited (Linear Probing):", average_nodes_visited_linear)
print("Average Nodes Visited (Quadratic Probing c1=0.5, c2=0.5):", average_nodes_visited_quadratic_half)
print("Average Nodes Visited (Quadratic Probing c1=0, c2=1):", average_nodes_visited_quadratic_one)


# In[21]:


empty_spaces_linear = len([slot for slot in linear_probing_hash_set.table if slot is None])
print("Empty spaces in Linear Probing:", empty_spaces_linear)
empty_spaces_quadratic_half = len([slot for slot in quadratic_probing_hash_set_half.table if slot is None])
print("Empty spaces in Quadratic Probing (c1=0.5, c2=0.5):", empty_spaces_quadratic_half)
empty_spaces_quadratic_one = len([slot for slot in quadratic_probing_hash_set_one.table if slot is None])
print("Empty spaces in Quadratic Probing (c1=0, c2=1):", empty_spaces_quadratic_one)


# In[24]:


import unittest

class TestChainingHashSet(unittest.TestCase):
    
    def test_insert_and_search(self):
        hash_set = ChainingHashSet()
        hash_set.add("test")
        self.assertTrue(hash_set.contains("test"))
    
    def test_length(self):
        hash_set = ChainingHashSet()
        hash_set.add("test")
        self.assertEqual(len(hash_set), 1)

    def test_insertion_and_retrieval(self):
        hash_set = ChainingHashSet()  # Replace with the appropriate class
        hash_set.add("apple")
        self.assertTrue(hash_set.contains("apple"))


    # Add more tests as needed
class TestLinearProbingHashSet(unittest.TestCase):
    
    def test_insert_and_search(self):
        hash_set = LinearProbingHashSet()
        hash_set.add("test")
        self.assertTrue(hash_set.contains("test"))
    
    def test_length(self):
        hash_set = LinearProbingHashSet()
        hash_set.add("test")
        self.assertEqual(len(hash_set), 1)

    def test_insertion_and_retrieval(self):
        hash_set = LinearProbingHashSet()
        hash_set.add("apple")
        self.assertTrue(hash_set.contains("apple"))
        
class TestQuadraticProbingHashSet(unittest.TestCase):
    
    def test_insert_and_search(self):
        quadratic_probing_hash_set_half = QuadraticProbingHashSet(c1=0.5, c2=0.5)
        quadratic_probing_hash_set_one = QuadraticProbingHashSet(c1=0, c2=1)
        quadratic_probing_hash_set_half.add("test")
        quadratic_probing_hash_set_one.add("test")
        self.assertTrue(quadratic_probing_hash_set_half.contains("test"))
        self.assertTrue(quadratic_probing_hash_set_one.contains("test"))
    
    def test_length(self):
        quadratic_probing_hash_set_half = QuadraticProbingHashSet(c1=0.5, c2=0.5)
        quadratic_probing_hash_set_one = QuadraticProbingHashSet(c1=0, c2=1)
        quadratic_probing_hash_set_half.add("test")
        quadratic_probing_hash_set_one.add("test")
        self.assertEqual(len(quadratic_probing_hash_set_half), 1)
        self.assertEqual(len(quadratic_probing_hash_set_one), 1)
        
    def test_insertion_and_retrieval(self):
        quadratic_probing_hash_set_half = QuadraticProbingHashSet(c1=0.5, c2=0.5)
        quadratic_probing_hash_set_one = QuadraticProbingHashSet(c1=0, c2=1)
        quadratic_probing_hash_set_half.add("apple")
        quadratic_probing_hash_set_one.add("apple")
        self.assertTrue(quadratic_probing_hash_set_half.contains("apple"))
        self.assertTrue(quadratic_probing_hash_set_one.contains("apple"))


# In[25]:


if __name__ == '__main__':
    unittest.main()


# In[ ]:




