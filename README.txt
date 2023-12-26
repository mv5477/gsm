------------
GSM
------------

Data scrapping and processing, into embeddings learning, applied to professional tennis results.

For the high-level documentation, including motivations and model details:
=> check ./doc/notes.odt

For the low-level documentation (code):
=> check ./doc/apidoc/_build/html/index.html


Usage:
- pip install requirements.txt
- unit tests: py -m unittest (from project root folder)
- data collection: py -m gsm.run_webscrapping (create './data/players' and './data/results' folders at the project root path beforehand)
- data preparation: py -m gsm.run_data_prep
- stats on gathered data: py -m gsm.run_stats
- learning and test: py -m gsm.run_player2vec (create './data/results_test' and put some data folders in there to be used as test cases)
