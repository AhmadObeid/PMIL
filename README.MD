How to run the scripts in data_preperation.py:
1) Run <dataset>/preprocess.py, specifically get_bad(), to generate unwanted_imgs.txt
2) Run running.py, which runs generate_persistence.py, and also generates extra_unwanted.txt
3) Run <dataset>/preprocess.py, specifically eliminate_bad() to get rid of unwanted imgs