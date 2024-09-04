# Harmful-Content-Detection
Industrial Machine Learning M9-BKK-2024

## Dataset
Detecting racism and xenophobia using deep learning models on Twitter data: CNN, LSTM and BERT Latest (zenodo.org)


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

# 5. Conclusion
The dataset we used is annotated from [this research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9044360/). We preprocessed it into a usable format. The dataset was analyzed and some insights about token distribution were found. We then explored two different methods of vectorizing sentences, including TF-IDF and embeddings, and found that embeddings give better performance in general because of their ability to capture the semantics of languages. From model exploration, we found that using logistic regression and random forest on sentence embedding already had both a great performance of 79% accuracy and a small model size. While LSTM and transformers, theoretically, should have better performance because of their ability to make use of sequential information in sentences, they gave similar performance but with more complex architecture and more resource consumption in our experiment. Overall performance should be improved with a larger dataset, especially of neural network models. 

# 6. Future Work
### Docker Containerization: 
In the next phase of development, we plan to containerize our machine-learning experiments using Docker. By encapsulating our experiments in Docker images, we aim to provide a portable and reproducible environment for users to easily deploy and run the models on various platforms. This also offers us the opportunity to have a demonstration for the users of how our model performs.

### Testing and Validation:
We will implement comprehensive testing procedures to ensure the model's performance under diverse scenarios. This includes unit tests for individual components and integration tests for the complete system.

### MLflow Integration:
In the next part, we plan to integrate MLflow into our system to enhance the overall management of machine learning models. This integration will systematically track and compare model experiments, enable versioning for easy traceability and reproducibility, establish a centralized model registry for collaborative development, and extend support for model serving, ensuring a seamless transition from experimentation to deployment. By leveraging MLflow's capabilities, our goal is to create a more organized, transparent, and deployable machine learning workflow.
