# Real-Fake-Job-Posting

## Data:
Real-Fake Job Posting Classification (NLP) from Kaggle: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction

I have been developing an NLP, multi-label classification model for a work-related project and thought I might practice and test my understanding by working through a binary classification model. This dataset is directly pulled from Kaggle (link above) and contains 17 880 job postings, of which, only 688 (~4.8%) are fraudulent. **(Insert Image)**

## Exploring the Data:
To simplify my analysis, I divide the categorical features into two main categories:
  1. Long unique descriptors: `company_profile`, `description`, `requirements`...
  2. Categorical indicators that are more receptive to one-hot-encoding (or are already binary): `has_questions`, `telecommuting`, `employment_type`...
  
A cursory analysis of the categorical indicators reveals that most either have too many different categories (> 20 categories), or have too many missing values to have much predictive value. I decide that employment type (6 classes, 19% missing values) can be included: **Insert Image**

Intuitively, one might think salary information might be an indicator of fraud. `salary_range` however, has a wide range of rates and currencies. A binary variable of whether or not salary data is included might be more informative: **Insert Image**

## Basic Feature Engineering & Baseline Models:
Before diving into NLP deep learning, I want to create a baseline model using existing categorical variables along with basic featuers we can extrapolate from the long unique descriptors. I decide to use `description`, the job description, as the main driver for new features. Simple numerical features I can extract from a description include: `num_nums` `num_punct` `description_length` and `num_links`

The dataset is an imbalanced dataset. To correct for this, I give fraudulent posts higher improtance in my three baseline models (logistic, random forest, XGBoost): **insert image**


The RF and XGBoost results have ROC AUC scores clsoe to 90%. This is a good starting point. Let's see if we can improve with deep learning methods. 

## NLP Preprocessing 
To prepare unstructured text for deep learning methods. We'll need to apply several preprocessing steps to our description data:
  1. standardize text to lower case, removing numerical data and extra white spaces
  2. removing stop words (ie. the, is) and short words (I choose to remove words less than 3 characters long)
  3. translating foreign descriptions (using `langdetect` & `googletrans` packages)
  4. to reduce the feature space, I lemmatize words (ie. children --> child). I use `SpaCy` lemmatizer after tagging words with their POS. 

`SpaCy` comes with its own built-in deep learning text categorizer that leverage CNNs. Nonetheless, the lack of customizability makes it difficult to implement. We will stick with TF/Keras
  
