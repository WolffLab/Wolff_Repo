# About Wolff_Data_Utils:

__DataHolder.py__: 
Takes in the raw press.csv and session.csv data from each rat and does preprocessing on it. 

__DataHolder_Plot.py__: 
Takes in objects produced by the DataHolder class and creates plots (IPI, CV, SR, etc.) 

__DataAvgs.py__:
Takes in multiple DataHolder objects and averages them by group. 

__DataAvgs_Plot.py__: 
Takes in objects produced by the DataAvgs class and creates plots (IPI, CV, SR, etc.) 

__init__.py, process_csv.py, and simple.py__: 
Referenced in the above files, each does behind-the-scenes work. 
