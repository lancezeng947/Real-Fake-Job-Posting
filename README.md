# Real-Fake-Job-Posting

## Data:
Real-Fake Job Posting Classification (NLP) from Kaggle: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction

I have been developing an NLP, multi-label classification model for a work-related project and thought I might practice and test my understanding by working through a binary classification model. This dataset is directly pulled from Kaggle (link above) and contains 17 880 job postings, of which, only 688 (~4.8%) are fraudulent. 

![labels](/Images/label_dist.JPG)

## Exploring the Data:
To simplify my analysis, I divide the categorical features into two main categories:
  1. Long unique descriptors: `company_profile`, `description`, `requirements`...
  2. Categorical indicators that are more receptive to one-hot-encoding (or are already binary): `has_questions`, `telecommuting`, `employment_type`...
  
A cursory analysis of the categorical indicators reveals that most either have too many different categories (> 20 categories), or have too many missing values to have much predictive value. I decide that employment type (6 classes, 19% missing values) can be included: 

![labels](/Images/employ_type_dist.JPG)

Intuitively, one might think salary information might be an indicator of fraud. `salary_range` however, has a wide range of rates and currencies. A binary variable of whether or not salary data is included might be more informative: 

![labels](/Images/salary_dist.JPG)

## Basic Feature Engineering & Baseline Models:
Before diving into NLP deep learning, I want to create a baseline model using existing categorical variables along with basic featuers I  extrapolate from the long unique descriptors. I decide to use `description`, the job description, as the main driver for new features. Simple numerical features I can extract from a description include: `num_nums` `num_punct` `description_length` and `num_links`

The dataset is an imbalanced dataset. To correct for this, I give fraudulent posts higher improtance in my three baseline models (logistic, random forest, XGBoost): ![labels](/Images/Baseline_metrics.JPG)


I use ROC AUC as the target performance metric. In the case of fraud detection, it can be argued either way on whether you want better **precision** (jobs flagged are actually fraudulent) or better **recall** (accurately flag all fraudulent jobs postings). As a potential job site, hosting these job postings, you do not want to frustrate the employers by suggesting their job post is fake, but at the same time you do not want your customers wasting their time applying to fake jobs. The ROC AUC metric will measure the performance of the overall balance between these two metrics.   

The RF and XGBoost results have ROC AUC scores clsoe to 90%. This is a good starting point. Let's see if we can improve with deep learning methods. 

## NLP Preprocessing 
To prepare unstructured text for deep learning methods. I apply several preprocessing steps to our description data:
  1. standardize text to lower case, removing numerical data and extra white spaces
  2. removing stop words (ie. the, is) and short words (I choose to remove words less than 3 characters long)
  3. translating foreign descriptions (using `langdetect` & `googletrans` packages)
  4. to reduce the feature space, I lemmatize words (ie. children --> child). I use `SpaCy` lemmatizer after tagging words with their POS. 

A cursory analysis look at the top 15 words used in fraudulent vs. clean descriptors, indicate relative similarity, with 11 words overlapping. A quick look at the lexical diversity of the descriptions also indicates relatively little difference (for more information on lexical diversity: https://pypi.org/project/lexical-diversity/. Nonetheless, let's see if deep learning algorithms can give us a little more insight. 

## Deep Learning NLP:
My NLP deep learning models will consist of three stages:
  1. Models using condensed sentence embeddings
  2. Sequence LSTM/GRU architecture models using individual word embeddings
  3. Blended models combining Sequence models with categorical features
  
**1.** To build a sentence embedding model, I first convert our descriptions into word embeddings. I choose to use gensim's preloaded FastText model `fasttext-wiki-news-subwords-300` With more time or resources I might decide to use the full FastText common crawl model, which can generate word embeddings for words outside of its vocabulary: https://fasttext.cc/docs/en/english-vectors.html. In our current situation, misspelled or unrecognized words will have an embedding vector of zeros. 

To convert word embeddings to sentence embeddings, I use a smooth-inverse-frequency aggregation method implemented by the `fse` package: https://github.com/oborchers/Fast_Sentence_Embeddings, based on research at Princeton: https://github.com/PrincetonML/SIF

I use these 300x1 as input features into our three (log, RF, XGB, 2-layer MLP) models, with similar weighting grid searches. Here are the results: [0,0]: LOG | [0,1]: RF | [1,0]: XGB | [1,1]: MLP

![labels](/Images/SIF_log.JPG)
![labels](/Images/SIF_RF.JPG)
![labels](/Images/SIF_XGB.JPG)
![labels](/Images/SIF_MLP.JPG)


Using sentence embeddings, the random forest classifier performed the best: 95% AUC. Unfortunately, the recall @ 45% is a bit low (ie. we are capaturing less than half the fake jobs posts)

**2.** Sentence embedding performed admirably, but by condensing an order-dependent sequence of words into a single representation, lots of information gets lost. The LSTM/GRU model will incorporate word embeddings as individual features and also extract contextual information from the order of words. Inspiartion for my code came from MLWhiz: https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/

The architecture of this model consists of an embedding layer (using pretrained FastText weights as starting points), followed by a bidirectional LSTM. We will pool results together to streamline the training process before outputing a sigmoid probability of whether or not the job posting is fake. The result outperforms all other models so far:
![labels](/Images/LSTM_CM.JPG)


**3.** TO DO...









  
