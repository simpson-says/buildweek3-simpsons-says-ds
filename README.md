# buildweek3-simpsons-says-ds
This is the Data Science subrepository.

The `.ipnyb` files contain the notebooks where the original data processing was developed. 

* `Simpions_says.ipynb` contains the search function that we used to match a user query to the most similar show quotes.
* `Simpsons_Writes_V4.ipynb` Contains the recurrent neural network that was used to generate synthetic dialogue for particular characters.
* `app.py` contains the Flask web app that empowers our website.  Most of the remaining files are pickled parts of our NLP model that must be loaded into this file.