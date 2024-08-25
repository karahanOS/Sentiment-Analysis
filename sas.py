import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Use the same model and tokenizer for consistency
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Load the reviews dataset
df_reviews = pd.read_csv(r'C:\Users\Karahan C\Desktop\Portfolio Projects\MARKETING\CSV FILES\FACT_Reviews.csv')

# Function to get sentiment score from review text
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt',clean_up_tokenization_spaces=True)
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

# Apply the sentiment score function to the review text column
df_reviews['sas'] = df_reviews['ReviewText'].apply(lambda x: sentiment_score(x))





# df_reviews.to_csv(r'C:\Users\Karahan C\Desktop\Portfolio Projects\MARKETING\CSV FILES\Processed_Reviews.csv', index=False)

print(df_reviews.head())
