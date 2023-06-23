# install.packages('lsa')
# library('lsa')

# Language Modeling - Word2Vec 
# Missing word prediction
# Next word prediction
# Words follow a particular sequence

# word2vec - Sparse Neural Network model (1 single hidden layer)
# Input: Raw text. (wikipedia)
# Output: Next word (similar words - Contextual words)

# Thou shalt not make a machine in the ilkeness of human mind

# CBOW() -  Continuous Bag of Words Representation
# Skipgram - Representation (more popular)

# Co-occurence matrix
# Thou (i/p) - Not(o/p)
# shalt (i/p) - Not(o/p)
# make(i/p) - Not(o/p)
# mind (i/p) - Not(o/p)

# Similar Vector Representation of a input gram(word)
# Neural Network need 3 things(optimizer, activation, loss function)
# Softmax function - assign probabilities to words 
# (we will pick the words with max probability) - sigmoid activation
# weights are called embedded matrix - word embedding.The input weight matrix
# supervised learning approach

# simple neural network - use RELU activation
# what is the loss? 
# Categorical_Crossentropy # More than 2 categories of output possible (more than 2 different possibilities)
# Binary_crossentropy # If only 2 categories
# If one unit, should use MAE,MSE

# Need to pass some incorrect input as well
# For each positive sample, take 2 negative samples

# Load Libraries
library(stringr)
library(tidyr)
library(dplyr)
library(tm)
library(keras) # wrapper for tensorflow
library(tensorflow)
library(purrr)
library(text2vec)
# Read Dataset
IMDB_Dataset <- read.csv("IMDB Dataset.csv")


# Text Processing
IMDB_Dataset <- IMDB_Dataset %>%
  mutate(review = str_remove_all(review, "<br />")) 

text_corpus <- VectorSource(IMDB_Dataset$review)
corpus <- Corpus(text_corpus)

text_corpus <- tm_map(corpus, tolower)
text_corpus <- tm_map(corpus, removeNumbers)
text_corpus <- tm_map(corpus, removePunctuation)
text_corpus <- tm_map(corpus, removeWords, stopwords(kind = "en"))
# text_corpus <- tm_map(corpus, stripWhitespace)

IMDB_Dataset$review <- corpus$content
IMDB_Dataset$review <- str_squish(IMDB_Dataset$review)

# Prepare data for Neural Network Model
# LDA DTM or TDM are inputs for machine learning models (TF-IDF)
# Neural Network provides the weigh to us, help us fix those weighs.
# Words should be converted to numerical format 
# Each word would be converted to a number (or index)
tokenizer <- text_tokenizer(num_words= 20000)
tokenizer  <- fit_text_tokenizer(tokenizer,
                                 IMDB_Dataset$review)
word_indexes <- tokenizer$word_index

# Create a skip-gram function
skipgrams_generator <- function(text, tokenizer, window_size, negative_samples) {
  gen <- texts_to_sequences_generator(tokenizer, sample(text))
  function() {
    skip <- generator_next(gen) %>%
      skipgrams(
        vocabulary_size = tokenizer$num_words, 
        window_size = window_size, 
        negative_samples = 1
      )
    x <- transpose(skip$couples) %>% map(. %>% unlist %>% as.matrix(ncol = 1))
    y <- skip$labels %>% as.matrix(ncol = 1)
    list(x, y)
  }
}

# Use a single hidden layer (Neural Network extracts hidden the layer)
embedding_size <- 300
skip_window <- 5 # Number of n-grams to use
num_sampled <- 1 # Number of negative sample to use for each correct word


# Create model architecture
input_target <- layer_input(shape = 1) # input
input_context <- layer_input(shape =1) # output

embedding <- layer_embedding(
  input_dim = tokenizer$num_words + 1, # treated as unknowns
  output_dim = embedding_size, 
  input_length = 1, 
  name = "embedding"
)

target_vector <- input_target %>% 
  embedding() %>% 
  layer_flatten()

context_vector <- input_context %>%
  embedding() %>%
  layer_flatten()

dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)
output <- layer_dense(dot_product, units = 1, activation = "sigmoid")

model <- keras_model(list(input_target, input_context), output)
model %>% compile(loss = "binary_crossentropy", optimizer = "adam")

summary(model)

model %>%
  fit_generator(
    skipgrams_generator(IMDB_Dataset$review, tokenizer, skip_window, num_sampled), 
    steps_per_epoch = 1000, epochs = 5,
    verbose = 1
  )

embedding_matrix <- get_weights(model)[[1]]
row.names(embedding_matrix) <- c("UNK", names(tokenizer$word_index)[1:20000])

find_similar_words <- function(word, embedding_matrix, n = 5) {
  similarities <- embedding_matrix[word, , drop = FALSE] %>%
    sim2(embedding_matrix, y = ., method = "cosine")
  
  similarities[,1] %>% sort(decreasing = TRUE) %>% head(n)
}
find_similar_words("suspense", embedding_matrix, n=20)

# Sentiment Analysis
# Machine Learning Model using neural networks
# 1. Create training and testing samples
# 2. Create model
# 3. Do model fitting

set.seed(1234)
train_test_sample <- sample(c(0,1), size = 50000, 
                            replace = T, 
                            prob = c(0.8,0.2))

table(train_test_sample)

x_train <- IMDB_Dataset[train_test_sample ==0, ]
x_test <- IMDB_Dataset[train_test_sample ==1, ]
y_train <- ifelse(x_train$sentiment == "positive", 1, 0)
y_test <- ifelse(x_test$sentiment == "positive", 1, 0)

# Input data for our neural network
tokenizer <- text_tokenizer(210000)
tokenizer_train <- tokenizer %>%
  fit_text_tokenizer(x_train$review)

tokenizer_test <- tokenizer %>%
  fit_text_tokenizer(x_test$review)

# without using texts_to_sequences, we don't have idea how to differentiate each review
tokenizer_train <- texts_to_sequences(tokenizer_train, x_train$review)
tokenizer_test <- texts_to_sequences(tokenizer_test, x_test$review)

# For example. on an average, all the reviews are of 125 words in length. convert everything into one level. 
# pad_sequences with max_length condition.
sequence_input_train <- pad_sequences(tokenizer_train,
                                      maxlen = 125)

sequence_input_test <- pad_sequences(tokenizer_test,
                                     maxlen = 125)

# Input is 40054*125 dimension for keras_model_dimension
# What is the dimension of word2vec embedding layer?
# It should be the number of tokens considered the feature dimension we ask for
# Input tokens we are considering
num_tokens <- length(unique(tokenizer$word_index))
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_tokens + 1,
                  output_dim = 300, # expecting embedding matrix of 300 dims
                  input_length = 125 # Max length of words in each review
  ) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model %>% keras::fit(
  sequence_input_train, y_train,
  epochs = 10, 
  batch_size = 500
)

# Test the model performance
model %>%
  evaluate(sequence_input_test,
           y_test)

# Make predictions a create a confusion matrix
predicted_sentiment <- model %>%
  keras::predict(sequence_input_test)

# look at embedding matrix weights only
model_weights <- get_weights(model)


