after putting this file where you want it. Make the following changes:

in activate_process_csv.py:
    on line 12 change "C:\\Program Files\\MATLAB\\R2021b\\bin\\matlab.exe\" to the path to your matlab executable, it may be the same

in process_mat.bat:
    on line 2 change to the path to your anaconda executable
    on line 3, if you want to work in a specific environement then change it to call that environement, otherwise delete that line


Usage:
    double click on the process_mat.bat file within this folder
    navigate to the .mat file you want to convert and select it
    two csv files will populate within the folder, one has the session info and one has the press info

