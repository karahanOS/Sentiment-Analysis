import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Use the same model and tokenizer for consistency
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Load the reviews dataset
df_reviews = pd.read_csv(r'C:\Users\Karahan C\Desktop\Portfolio Projects\MARKETING\CSV FILES\FACT_Reviews.csv')
df_reviews.info()           
df_reviews.describe()
print(df_reviews.head())



# Function to get sentiment score from review text
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt', max_length=1024, truncation=True)
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

# Apply the sentiment score function to the review text column
df_reviews['sas'] = df_reviews['ReviewText'].apply(lambda x: sentiment_score(x))


# df_reviews.to_csv(r'C:\Users\Karahan C\Desktop\Portfolio Projects\MARKETING\CSV FILES\Processed_Reviews.csv', index=False)

# sns.pairplot(data=df_reviews, vars=['Rating', 'sas'], hue='sas')
# plt.show()            

print(df_reviews.query('Rating==1').sort_values('sas', ascending=False)['ReviewText'].values[0])          
