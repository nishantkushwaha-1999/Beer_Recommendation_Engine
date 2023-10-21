# Beer Recommendation Engine
This data science project aims develop a beer recommendation engine utilizing the power of sentiment analysis using LLMs. This engine aims to capture the likeness of a beer attribute from general conversations on an online forum.

## Dataset
The dataset used in this project is the comments and replies from the forum - https://www.beeradvocate.com/beer/top-rated/. The dataset has been preprocessed and cleaned for use in this project.

## Methodology
The engines tries to make use of simmilarity scores between the selected attributes (provided to the user) and the comments. The comments with a high similarity score and positive sentiment are used for recommendation. The similarity scores and the sentiment scores are combined using the formula - 

$$ final_score = log₂(sentiment) * log₂.₅(similarity) $$

The final_scores are aggregated to the product level and a new_final_score is calculated using the avg_final_score and #comments (after all the filters) as - 

$$ new_final_score = log(no. of comments) * (log₂(avg_final_score)/log(1.5)) $$

Finally top three beers are recommended based on the new_final_score.

## Code
The code for this project is written in Python 3 and is contained in the Final Submission/Final Notebook_V3.0.ipynb.ipynb Jupyter notebook. The notebook uses several popular data analysis libraries such as pandas, numpy, scikit-learn, nltk, openai, spacy and selenium.

The notebook is divided into several sections, including data loading and preprocessing, exploratory data analysis, and model building. Each section includes detailed explanations of the code and its functionality.

## Results
The analysis also aims at finding out the nuances of using word-embeddings instead of bag-of-words for calculating similarity between attribute and comments in NLP. Also, the efficacy of LLMs over VADER in detecting human sarcasm through sentiment analysis using OpenAI is explores in this analysis.

The results and explanation are all present in the same notebook.

## Conclusion
This project unravels some unseen insights in this segment and can be read at link
