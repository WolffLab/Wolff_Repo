import pandas as pd
import numpy as np
from pandas import NA as nan
import sys,os
import h5py
from Wolff_Repo.Utils.simple import *
import json

class traj_parser:


    def __init__(self, parser_dir, traj_dir = None):
        """Initialize traj_parser class
        
        Parameters
        ----------
        parser_dir : str
            Path with h5 file and config.json
        traj_dir : str, optional
            Path to directory with trajectory h5 files. 
            You don't need to include a value for this, however a value must be defined using self.vidloc = path before self.parse_vids() will work.

        Returns
        -------
        out : traj_parser"""
        self.configloc = parser_dir + r'\config.json'
        self.init_configfile()
        self.h5loc = parser_dir + f"\\{self.configfile['Subject']}.h5"

        if not isinstance(traj_dir,type(None)):
            self.vidloc = traj_dir

    @property
    def vidloc(self):
        return self._vidloc

    @vidloc.setter
    def vidloc(self,val):
        self._vidloc = val

    def update_configfile(self):
        """updates configuration .json file to match self.configfile"""
        with open(self.configloc,'w') as jsonfile:
            json.dump(self.configfile,jsonfile,indent = 4)
    
    def init_configfile(self):
        """Uses the path 'configloc' taken from init to pull the json file, after this function is ran you can call configfile and use it like a dictionary"""
        with open(self.configloc,'r') as jsonfile:
            self.configfile = json.load(jsonfile)
        if not 'Parsed_Traj_Files' in self.configfile:
            self.configfile['Parsed_Traj_Files'] = []
            self.update_configfile()

    def determine_filelist(self):
        """Determine the files in the directory that need to be parsed through. Files that have already been parsed are not included in the list."""
        self.filelist = []
        for path in os.listdir(self.vidloc):
            #check if file is an h5 file
            if path.split('.')[-1] == 'h5':
                #check if file has already been parsed
                if not path in self.configfile['Parsed_Traj_Files']:
                    self.filelist.append(path)
    
    def parse_vids(self):
        """Go through the directory of video data as h5 files and store in the main h5 file for that rat.
         The path to the directory of the video files must have been previously defined via self.vidloc = val."""
        self.determine_filelist()
        nmax = len(self.filelist)
        n = 0

        with h5py.File(self.h5loc,'r+') as f:
            for sub_path in self.filelist:
                n+=1
                print(f'parsing file {n}/{nmax}',end = '\r')
                path_vid = self.vidloc + '\\' + sub_path
                #get session number
                sess = take_inner(sub_path, strtstr = 'Sess_', endstr = '_')
                #get trial number
                trial = take_inner(sub_path, strtstr = 'Trial_', endstr = '.')
                if '_' in trial:
                    trial = take_inner(trial, endstr= '_')
                #get cam side
                side = sub_path.split('_')[0]
                #get the subfolder for the session within the h5 file
                sess_file_name = list(f.keys())[str_in_list(f.keys(),digitnum(sess,4))]
                sess_file = f[sess_file_name]

                #create a trajectories subfolder for that particular side if it does not yet exist
                traj_folder_name = side + '_trajectories'
                if not traj_folder_name in sess_file.keys():
                    sess_file.create_group(traj_folder_name)
                with h5py.File(path_vid,'r') as f_2:
                    #this is a list of the bodyparts specified by each column in the trajectories table
                    bodyparts = f_2['node_names'][:].astype(str)
                    #the datatype will establish column names in resulting table
                    ds_dtp = np.dtype([(j,[(k, np.float32) for k in ['x', 'y']]) for j in bodyparts])
                    #a rec array is a convinient datatype for creating the table in the h5 file
                    traj_rec = np.rec.array(list(totuple(f_2['tracks'][0].T)), dtype = ds_dtp)
                    dset_name = 'Trial_' + digitnum(trial,4)
                    sess_file[traj_folder_name].create_dataset(dset_name, data = traj_rec)
                    sess_file[traj_folder_name][dset_name].attrs['file'] = sub_path
                self.configfile['Parsed_Traj_Files'].append(sub_path)
                self.update_configfile()

    def store_paths(self, wipe = False):
        """Prompt user to select the path to directories that contain the trajectory h5 files within the config file for a rat.
        Store these values for later use.
        
        Parameters
        ----------
        wipe : bool, optional
            If true this will wipe all previously stored paths."""
    
        #if wipe is true delete existing list of directories
        if wipe and "trajectory_dir" in self.configfile:
            del self.configfile['trajectory_dir']

        #create list if it does not already exist and add first entry
        if not "trajectory_dir" in self.configfile:
            traj_dir = get_directory("Select Directory of Trajectory Files")
            self.configfile['trajectory_dir'] = [traj_dir]

        #ask user if they want to add another entry, keep asking and adding till user says no 
        while ynprompt("Add another directory?"):
            self.configfile['trajectory_dir'].append(get_directory("Select Directory of Trajectory Files"))
        self.update_configfile()

        
    @classmethod
    def parse_trajectories(cls, parser_dir):
        """Update an h5 file with trajectory data.
         If this is the first time then function will prompt user to select paths to directories with trajectory files.
         These paths will be saved, paths can be changed later by directly modifying json files.

        
        Parameters
        ----------
        cls : traj_parser
        parser_dir : str
            Path to directory with [subject].h5 and config.json file"""

        tempclass = cls(parser_dir)
        if not "trajectory_dir" in tempclass.configfile:
            tempclass.store_paths()
        for dir in tempclass.configfile['trajectory_dir']:
            tempclass.vidloc = dir
            print(f'parsing {dir} for new trajectory files.')
            tempclass.parse_vids()

    @classmethod
    def setup_dir(cls, parser_dir):
        """Call `store_paths` on a specific directory. Only runs if trajectory_dir has not already been defined in the configfile.
        
        Parameters
        ----------
        parser_dir : str
            Path to directory."""
        tempclass = cls(parser_dir)
        if not 'trajectory_dir' in tempclass.configfile:
            tempclass.store_paths()
