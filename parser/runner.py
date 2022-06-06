#a function to run the parser on a directory
from new_dir_setup import new_dir_setup
from event_parser_hdf5 import event_parser_hdf5
from traj_parser import traj_parser
from Wolff_Repo.Utils.simple import *
import os

if ynprompt('Use text file? (y/n)'):
    datdir = open(get_file()).read().split('\n')
else:

    #prompt user to select a directory
    datdir = [get_directory().replace('/','\\')]

include_trajectories = ynprompt('Would you like to include trajectories? (y/n)')

for dir in datdir:
    print(f'setting up {dir}')
    #check if this directory has already been setup by seeing if it has the parser subfolder
    if not 'parser' in os.listdir(dir):
        new_dir_setup.setup(dir)
    
    #set the directory where the h5 file is located
    active_dir = dir + r'\parser'

    #add paths to trajectory files for each rat
    if include_trajectories:
        traj_parser.setup_dir(active_dir)

#loop through each directory and actually run parser
for dir in datdir:
    print(f'parsing {dir}')

    #set the directory where the h5 file is located
    active_dir = dir + r'\parser'

    #run the parser
    event_parser_hdf5.parsedirectory(active_dir, dir)
    if include_trajectories:
        traj_parser.parse_trajectories(active_dir)
