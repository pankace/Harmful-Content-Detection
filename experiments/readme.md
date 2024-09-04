
## Detecting Racism Experiments
   **Team name:** Echo B.
   
   **Team:** Tommaso, Philip, Valentine, Pankace

# 1. Introduction

This project takes on the challenge of identifying racist content in tweets through a series of machine-learning experiments. We performed data analysis to better understand the dataset, and we experimented with text representation methods like TF-IDF and embedding, and we evaluated various models, including logistic regression, random forest, LSTM, and BERT to develop an effective approach for classifying tweets.

# 2. Data Overview

Our experiments are based on a dataset of 2,000 tweets, each labelled as either racist (1) or not racist (0).

# 3. Data Sample Analysis and Validation

## 3.1 Data Preprocessing and Validation

The data preprocessing is vital and was done to reduce noise interference in the training process. We did this by analyzing the tweets and modifying them to have no emojis, punctuation, no website links, no “@mentions” and other elements that can be considered noise interference anywhere in the post-processed data.
The dataset is then validated to make sure that it is ready to be used.

### Missing Values:
Our examination revealed no missing values in the dataset, indicating that it is complete and ready for further processing

### Label Distribution:
The label distribution is nearly balanced, with 51.35% of negative labels (not containing racist content) and 48.65% of positive labels (containing racist content). This balance is beneficial for model training, as it reduces the risk of bias towards one class.


## 3.2 Exploratory Data Analysis (EDA)

### Token analysis:
we discovered that there were 59,875 total amount of tokens(after most of the noise interference was removed) and only 7,589 tokens were unique. This adds up to only 12.67% of the tokens being unique. 

### Word Frequency Analysis: 
To get a sense of the most common themes in the tweets, we conducted a word frequency analysis. These results highlight key topics and terms present in the dataset, with words like "immigrant," "boat," and "country" appearing frequently. This insight suggests that discussions around immigration are prominent in the sampled tweets, potentially pointing to areas of focus for detecting racist content.

### Sentence Length Analysis:
For a better understanding of the textual data we are working with and to inform both the preprocessing strategies and model choice. 

# 4. Model Exploration and Validation
Before we began model exploration, two methods of vectorizing tweets (representing sentences in numerical terms), including TF-IDF and embedding, were studied by clustering words in lower dimensions (after PCA) and analyzing each cluster.
 
### TF-IDF Clustering Insights:
The clusters formed from TF-IDF vectorization seem to revolve around explicit language use and identifiable themes related to immigration, legality, and potentially racially charged language. These clusters suggest that TF-IDF is good at capturing the presence of specific keywords and phrases that are strongly associated with the target categories within the dataset.

### Embedding Clustering Insights:
Clusters derived from embedding techniques, which capture more contextual information and semantic relationships between words, present a different picture. They seem to group tweets based on broader topics or narratives, which might not be as directly tied to the presence of specific keywords but rather to the overall meaning or context of the tweets.

### Overall Analysis:
The contrast between the clusters from TF-IDF and embeddings highlights the different strengths of each method. TF-IDF clusters are more keyword-focused, which can be beneficial for detecting explicit content but might miss subtler context-based nuances. On the other hand, embeddings provide a richer understanding of context and semantic relationships, which can be advantageous for identifying underlying themes and sentiments that are not as overt.


We carried out the following modelling experiments: Logistic regression, random forest, LSTM, transformers
```
Logistic Regression
with TF-IDF:
              precision    recall  f1-score  
           0       0.75      0.78      0.77       
           1       0.74      0.71      0.73 
Accuracy: The model achieved an accuracy of 75%.
Precision and Recall: balanced performance across both classes

with Embeddings:
              precision    recall  f1-score 
           0       0.80      0.79      0.80
           1       0.77      0.78      0.77
Accuracy: The model saw an improvement in accuracy to 78.5%
Precision and Recall: The precision and recall for both classes are slightly improved
```

From these results, it appears that the embeddings offer a modest improvement over TF-IDF vectors in terms of accuracy and other metrics. This could be because embeddings provide a more nuanced representation of the text data, capturing semantic relationships and context that TF-IDF may miss. Although the results are not overwhelmingly different therefore both methods might be viable depending on other factors.


### Random forest

Feature engineering: Each sentence had been lemmatized to its root form before their embedding were calculated
Training: Binary classifier using random forest.
Validation: The accuracy metric was 0.79. The precision metric was 0.7943. The recall metric was 0.7671. The F1 score was 0.7805. The processing time for this was 7.45 seconds.

### LSTM

Feature engineering: Each sentence had been lemmatized to its root form before they were tokenized by blank space. Each token was assigned a unique ID and its embedding is initialized. 
Training and Validation: The LSTM model is trained to output a single probability of a sentence being in a positive class. The best accuracy on a validation dataset is around 70%

### Transformers

Feature engineering: Because transformers are good at dealing with dependency between tokens, lemmatization wasn’t applied to avoid any loss of information. Only unnecessary artifacts such as emojis, punctuation, website links, Twitter handles, and HTML were removed.
Training and Validation: Distil-BERT is used to get embeddings of an input sentence and then a multi-layer perceptron classification head was trained to output probabilities between two classes. The best accuracy on a validation dataset is around  79%


# 5. Conclusion
The dataset we used is annotated from [this research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9044360/). We preprocessed it into a usable format. The dataset was analyzed and some insights about token distribution were found. We then explored two different methods of vectorizing sentences, including TF-IDF and embeddings, and found that embeddings give better performance in general because of their ability to capture the semantics of languages. From model exploration, we found that using logistic regression and random forest on sentence embedding already had both a great performance of 79% accuracy and a small model size. While LSTM and transformers, theoretically, should have better performance because of their ability to make use of sequential information in sentences, they gave similar performance but with more complex architecture and more resource consumption in our experiment. Overall performance should be improved with a larger dataset, especially of neural network models. 

 
# 6. Future Work
### Docker Containerization: 
In the next phase of development, we plan to containerize our machine-learning experiments using Docker. By encapsulating our experiments in Docker images, we aim to provide a portable and reproducible environment for users to easily deploy and run the models on various platforms. This also offers us the opportunity to have a demonstration for the users of how our model performs.

### Testing and Validation:
We will implement comprehensive testing procedures to ensure the model's performance under diverse scenarios. This includes unit tests for individual components and integration tests for the complete system.

### MLflow Integration:
In the next part, we plan to integrate MLflow into our system to enhance the overall management of machine learning models. This integration will systematically track and compare model experiments, enable versioning for easy traceability and reproducibility, establish a centralized model registry for collaborative development, and extend support for model serving, ensuring a seamless transition from experimentation to deployment. By leveraging MLflow's capabilities, our goal is to create a more organized, transparent, and deployable machine learning workflow.


