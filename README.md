# P-mean for fastText 
Sentence embedding model that use Power mean method. ref : [P-mean](https://arxiv.org/abs/1803.01400)  


## Requirements 

### Python version >= 3.5

### python bindings 
* gensim
* sklearn
* numpy

> pip install -r requirements.txt


## Example use cases

### get sentence vector from fastText
	import p_mean_FT as pmeanFT
	import fastText

	#fasttext model 
	fT=fastText.FastText.load_model("fasttext path")

	#what kind of power mean you use
	meanlist=['mean','p_mean_3','p_mean_2']

	#tokenized words
	words = ['i','like','you']
	sent_vec = pmeanFT.get_sentence_embedding( words, fT,meanlist) 	
	
### See BOW.ipynb for further model comparision 
* A model comparison between BOW model made of word2vec, word2vec pmean, fastText, fastText pmean

