import os
from Wolff_Repo.Utils.simple import get_file
from process_csv import conv_mat_to_csv

fullpath = get_file()
#location of file
loc=fullpath.split('/')[-1]
#name of file
path = fullpath.replace(loc,"")

#call matlab program to output a csv
os.system(f"call \"C:\\Program Files\\MATLAB\\R2021b\\bin\\matlab.exe\" -wait -nosplash -nodesktop -r \"loc='{loc}';path='{path}';run('mattocsv.m'); exit;\"")

conv_mat_to_csv(path,loc)