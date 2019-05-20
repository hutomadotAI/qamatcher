# Introduction 
The QA-Matcher receives the user queries and processes them to find the matching answer from the training file. It consists of two major parts: 

1. TrainingWorker 
2. ChatWorker 

Both of those classes call or prepare for 3 different matching methods: 
1. StringMatcher 
2. EntityMatcher 
3. EmbeddingMatcher 

All the parts mentioned above will be described in more detail in the following sections. 

## The Training Worker 
The training worker receives the training QA-pairs/ Intent utterances from ai_training, preprocesses them, extracts entities and fits a training model for the Embedding Matcher. 

For the 3 different matchers it needs to perform different tasks. For the Embedding-Matcher: 

  * Tokenizing the question and excluding the x-large set of stopwords as well as filtering certain entities 
  Query word2vec service for all word vectors for the individual words appearing in any training question 
  * Initializing the Embedding-Matcher with the relevant word vectors 
  * Calling the Embedding-Matcher to fit a training model 
  * Saving the model and word-vectors in a file 

For the String Matcher: 
  * Extract Custom Entities from the training questions (those are matched with Jon's entities during chat). This extraction is purely regex based and searching for the string between the custom entity identifiers "@{" and "}@". 
  * Tokenize the questions using different stopword sizes (large, small) and without filtering any entities 
  * Save the entities, tokenized questions and the raw Q&A's to a file 

For the Entity Matcher: 
  * Extract spacy entities from question and answers in training file 
  * Save those to a file 

## The Chat Worker 
The chat worker receives the user query from ai_training and finds the best matching training answer for it.  To do this it uses the above mentioned 3 different matching mechanisms. The steps are as follows: 

  * Tokenize user question using the x-large stopword set and filtering some entities 
  * Extract the spacy entities from the question 
  * Call the String Matcher (SM) to find the best possible match and corresponding score 
  * Call the Entity Matcher (EM) to find the best possible match and corresponding score 
  * If SM score > EM score and the SM score is larger than a threshold which is given in variable "STRING_PROBA_THRES" return the found answer from the String Matcher 
  * Otherwise take the result from EM if it's score is larger 0.0 
  * If both, SM and EM cannot find a satisfying match call the Embedding Matcher to find the answer 

This is the flow you find in method "chat_request" of the ChatWorker. When calling the method "get_string_match" in step 3, you find additional logic around the actual StringMatcher. This is necessary as the SM can find multiple matches with the same score.   

  * If the SM returns only 1 result this will be used and returned to the "chat_request" function 
  * If the SM return no match (not actually possible at the moment) the it will return and empty string with score 0.0 
  * If the SM returns more than 1 match then the EM is called with the subset of training questions which gave the same score in the SM 
  * If this call to EM only returns 1 match then we return this result with the EM score 
  * If more than one match is returned from EM we call the Embedding Matcher with the subset of training questions returned from SM and return this result with score EM + 0.1 
  * If any of the subset of training questions consists of only "UNK", which means all words in the training questions were stopwords, we return an empty string with score 0.0 
  
  When calling the method "get_entity_match" in step 4 there is similar logic in place. 
  
  When calling the method "get_embedding_match" in step 7 we go through the following steps: 
  * Find the unique tokens from the user query 
  * Check for tokens without a word embedding already in the EmbeddingComparison class 
  * Query the word2vec service if any of those words without an embedding do have an embedding in word2vec model 
  * If any, retrieve those embeddings from the word2vec service and add it to the EmbeddingComparison 
  * Call EmbeddingComparison model to find the best match with a training question/intent 
  * Return this match with the score reduced by 0.15 (this is done as the embedding comparison tends to be overly optimistic about its matches) 
 

# Build and Test
Use pipenv and pytest

# Contribute
TODO:
