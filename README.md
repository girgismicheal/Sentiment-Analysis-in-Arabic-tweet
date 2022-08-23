# Sentiment-Analysis-in-Arabic-tweet
You can open the notebook in Kaggle from the icon below.<br>
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/girgismicheal/sentiment-analysis-in-arabic-tweet/edit/run/103976227)

### **Introduction**

* **Natural Language Processing (NLP):** The discipline of computer science, artificial intelligence and linguistics that is concerned with the creation of computational models that process and understand natural language. These include: making the computer understand the semantic grouping of words (e.g. cat and dog are semantically more similar than cat and spoon), text to speech, language translation and many more

* **Sentiment Analysis:** It is the interpretation and classification of emotions (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis allows organizations to identify public sentiment towards certain words or topics.

![image](https://drive.google.com/uc?export=view&id=1KQkdq_eJ1dqIEnOIUdR3Gvi8uRe9pQqi)



# Table of Contents
1. [Environment Setup](#p1)
    - [Enable the GPU](#p1.1)
    - [Dependencies Installation](#p1.2)
2. [Dataset Importing](#p2)
3. [Dataset Preparation](#p3)
    - [Dataset Cleaning](#p3.1)
    - [Dataset Tokenization](#p3.2)
    - [Label Encoding](#p3.3)
    - [Train Test Spliting](#p3.4)
4. [Build Classical Machine learning Models](#p4)
    - [TF-IDF Embedding](#p4.1)
    - [Train Different Classifiers and Select the Champion Model](#p4.2)
5. [Tramsfer Learning a Pre-trained Models](#p5)
6. [Infer the test data and prepart the subbmion file](#p6)



# <a name="p1">Environment Setup</a>

### <a name="p1.1">Enable the GPU</a>
```Python
import torch
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    !nvidia-smi
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
```
### <a name="p1.2">Dependencies Installation</a>
```Python
!pip install gdown
!pip install pyarabic
!pip install farasapy
!pip install emoji
!pip install transformers
!git clone https://github.com/aub-mind/arabert.git
```


# <a name="p2">Dataset Importing</a>
The dataset has been scraped from Twitter and then labeled and used in a local competition in EGPYT, It contains 2,746 tweets extracted using the **Twitter API**. The tweets have been annotated (neg = negative, pos = positive, and neu = neutral) and they can be used to detect sentiment.

**Dataset files:**
- train.csv - the training set has 2059 unique entry
- test.csv - the test set has 687 unique entry


**The dataset has 2 fields:**
1. **Tweet**: the text of the tweet
2. **Class**: the polarity of the tweet **(neg = negative, pos = positive, and neu = neutral)**

![image](https://drive.google.com/uc?export=view&id=1f2RlQTR5QxLbOG4ygTeQZqeMjzZiTJV5)

```Python
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
Data_set = pd.read_csv("/kaggle/input/nlp-arabic-tweets/train.csv")
Data_set
```


