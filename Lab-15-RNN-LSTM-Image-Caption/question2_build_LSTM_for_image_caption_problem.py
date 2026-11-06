'''
Lab 15 - RNN Applications
Learning Goals
In this lab, you will learn commonly solved problems of Image captioning using RNN and LSTM
By the end of this lab, you should be able to
1. Build a RNN for image caption problem
2. Build Long Short-Term Memory (LSTM) model for image caption problem.Exercises

Exercises
2. Implement LSTM using PyTorch for image captioning

For the purpose of this lab the link of tutorial i followed .
tutorial: https://nishantprabhu.github.io/Image-Captioning-with-PyTorch-and-Keras/

Dataset
        : Flickr8k_Dataset.zip : https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
        : Flickr8k_text.zip : https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
'''
#---------------------------Imports---------------------------------#
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

import os
import string
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
#-------------------------------------------------------------------#

# Transforms, to get the image into the right format
img_transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
])

# Function to generate image features
def generate_image_features(image_dir):
    vgg = models.vgg16(pretrained=True)
    layers = list(vgg.features)[:-1]
    model = torch.nn.Sequential()

    features = {}

    for f in os.listdir(image_dir):
        path = image_dir + '/' + f
        image = Image.open(path)
        image = img_transform(image)
        image = image.unsqueeze(0)
        x = model(image)
        features.update({f: x})

    return features
# Generate the image features

image_features = generate_image_features("/home/ibab/DL-Lab/Data/Flickr8k_Dataset/Flicker8k_Dataset")

print(image_features)

with open("/home/ibab/DL-Lab/Data/Flickr8k_text/Flickr8k.lemma.token.txt") as f:
    captions = f.read().split("\n")
# Display the first 10 captions
print(captions[:10])

# Create dictionary of captions
captions_dict = {}
for line in captions:

    # Separate caption from filename
    contents = line.split("\t")

    # If there are no captions, then ignore
    if len(contents) < 2:
        continue
    filename, caption = contents[0], contents[1]
    # Exclude the number tag from filename
    filename = filename[:-2]

    # If the dictionary already has the filename in its keys ...
    # ....simply extend the list with current caption
    # Else, add the current captions in a list
    if filename in captions_dict.keys():
        captions_dict[filename].append(caption)
    else:
        captions_dict[filename] = [caption]

# Create lists of filenames that go into training and testing data
# We will do a 70-30 split
split_point = int(0.7 * len(captions_dict))
training_files = list(captions_dict.keys())[:split_point]
test_files = list(captions_dict.keys())[split_point:]

# Segregate image features and captions for train and test into separation
train_features, train_captions = {}, {}
test_features, test_captions = {}, {}

for name in training_files:
    train_features[name] = image_features[name]
    train_captions[name] = captions_dict[name]

for name in test_files :
    test_features[name] = image_features[name]
    test_captions[name] = captions_dict[name]

def preprocess_line(line):
    line = line.split()                     # Convert to a list of words
    line = [w.lower() for w in line]        # Convert to lowercase
    line = [w for w in line if w.isalpha()] # Remove numbers
    line = " ".join(line).translate(
        str.maketrans("","",string.punctuation)
    )
    line = "startseq " + line + " endseq"
    return line

def get_all_captions(captions_dist):
    """ Collects all captions in one list """
    captions = []
    for cpt_list in captions_dist.values():
        captions.extend(cpt_list)
    return captions

def tokenize_sentences(lines):
    tokenizer = Tokenizer(filters='') # Initialize the tokenization
    tokenizer.fit_on_texts(lines)     # This generates the word index map
    vocab_size = len(tokenizer.word_index) + 1 # The size of the vocabulary
    return tokenizer, vocab_size

def pad_tokens(tokens):
    return pad_sequences(tokens, padding='post') # Max length is auto-calculated

# clean all the captions
for filename, cpt_list in captions_dict.items():
    for i in range(len(cpt_list)):
        # Clean the caption
        cleaned_caption = preprocess_line(cpt_list[i])
        # Replace it in the correct location in the list
        cpt_list[i] = cleaned_caption

# Get all captions in a list and fit tokenizer

all_captions = get_all_captions(captions_dict)
tokenizer, vocab_size = tokenize_sentences(all_captions)
print("Size of the vocabulary:", vocab_size)





