from Wolff_Repo.Utils.simple import *
import pandas as pd
from zipfile import ZipFile
import numpy as np
import json
import sys
import os
import h5py
from pandas import NA as nan


class event_parser_hdf5:
    """
    Class used to parse through lists of events and write directly to HDF5"""
    #types of state machines to look for
    sess_types = ['TwoTapGeneral','TwoTapShort']
    #types of events to look for
    event_types = ['SMStarted','SMFinished','TwoTapUpper','TwoTapLower','TwoTapTarget','TwoTapData','GiveReward_Int','SoundSetting','LeverDI']
    #columns being tracked in dataframe
    sess_columns = ['Target','Upper','Lower','Start Time (Expt)','End Time (Expt)','Session Type','Trials in Session','File']

    #colums beings recorded about each tap as well as datatype info for that column
    tap_columns = {'Tap 1':{'dtype':[('On',np.uint32),('Off',np.uint32),('On_Expt',np.uint32)]},
    'Tap 2':{'dtype':[('On',np.uint32),('Off',np.uint32)]},
    'Interval':{'dtype':np.uint16},
    'Reward':{'dtype':[('Value',np.uint8),('Time',np.uint32)]},
    'Extra Taps':{'dtype':h5py.vlen_dtype(np.int32)},
    'N Taps':{'dtype':np.uint8},
    'Lick Time':{'dtype':np.uint32},
    'Tone Time':{'dtype':[('On',np.uint32),('Off',np.uint32)]}
    }

    #names of each column in the .dat files and a good datatype for them
    dat_columns = {'Event Type ID':'category',
    'Value':np.uint16,
    'Current Protocol':'category',
    'Current State':np.uint8,
    'Time Since State Machine':np.uint32,
    'Time Since Experiment':np.uint32,
    'Daily Water':np.uint16,
    'Weekly Water':np.uint16}
    #maximum amount of figures in session number (i.e if less than 10000 sessions expected chose 4, if less than 100000 chose five)
    max_figs = 4

    def __init__(self, active_dir, datdir):
        """Inititialize the event parser_class

        Parameters
        ----------
        active_dir : str
            path to the directory with the json configuiration file and data hdf5 file.
        datdir : str
            path to the directory where all .dat and .zip files are stored
        
        Returns
        -------
        out : event_parser
        """

        #store directory names in class
        self.datdir = datdir
        self.active_dir = active_dir
        self.configloc = active_dir + r'\config.json' 
        self.init_configfile()
        self.init_hdf5()

    @property
    def active_datfile(self):
        """Returns a dataframe made from the datfile that is currently being looked at.
        """
        return self._active_datfile

    @active_datfile.setter
    def active_datfile_path(self,loc):
        """Sets the path to the current active .dat file, loads file into a dataframe and optimizes datatypes.

        Parameters
        ----------
        loc : str
            path to the current active dat file.
        """
        #put information from dat file into a dataframe
        #make columns smallest possible datatype for efficiency using dtype argument
        try:
            datfile =  pd.read_csv(loc,names = self.dat_columns.keys(),
                                                dtype = self.dat_columns,
                                                usecols= ['Event Type ID',
                                                'Current Protocol',
                                                'Value',
                                                'Current State',
                                                'Time Since State Machine',
                                                'Time Since Experiment'])
        #this handler deals with the case when a csv row is missing some values
        except ValueError:
            #import csv without specifying datatypes so missing values are populated with na values
            datfile  = pd.read_csv(loc,names = self.dat_columns.keys(),
                                                usecols= ['Event Type ID',
                                                'Current Protocol',
                                                'Value',
                                                'Current State',
                                                'Time Since State Machine',
                                                'Time Since Experiment'])
            #delete na values
            datfile = datfile.dropna()
            #change column types now that there are no na values
            datfile = datfile.astype({j:self.dat_columns[j] for j in self.dat_columns.keys() if j in datfile.columns})
        
        #use state type dictionary to create column of state types
        datfile['State Name'] = [self.configfile['State_IDS'].get(i,'Not_Defined') for i in datfile['Current Protocol']]
        #set datatype to catagorical for optimization
        datfile['State Name'] = datfile['State Name'].astype('category')

        #use event type dictionary to create column of event types
        datfile['Event Type'] = [self.configfile['Event_IDS'].get(i,'Not_Defined') for i in datfile['Event Type ID']]
        #set datatype to catagorical for optimization
        datfile['Event Type'] = datfile['Event Type'].astype('category')

        datfile = datfile.loc[datfile['State Name'].isin(self.sess_types) & datfile['Event Type'].isin(self.event_types)]

        datfile.index = range(len(datfile))
        #store file in instance of self
        self._active_datfile = datfile

    def init_configfile(self):
        """Uses the path 'configloc' taken from init to pull the json file, after this function is ran you can call configfile and use it like a dictionary"""
        with open(self.configloc,'r') as jsonfile:
            self.configfile = json.load(jsonfile)
    
    def init_hdf5(self):
        """Create reference to HDF5 file that stores data within class. If the HDF5 file has already been started then create a reference to the most recent session."""
        self.h5loc = self.active_dir + fr"\{self.configfile['Subject']}.h5"
        self.h5 = h5py.File(self.h5loc,'r+')
        #check if the h5 file already has some sessions
        if self.h5.keys():
            #if the file already has some sessions then continue from where it left off
            self.dgroup =self.h5[list(self.h5.keys())[-1]]

    def update_configfile(self):
        """updates configuration .json file to match self.configfile"""
        with open(self.configloc,'w') as jsonfile:
            json.dump(self.configfile,jsonfile,indent = 4)

    def update_config_from_zip(self,path):
        """Updates the configuration .json file using information found in a zipfile

        Parameters
        ----------
        path : str
            path to the zipfile being used.
        """

        #pull the zipfile with configuration info
        myzip = ZipFile(path,'r')

        #decode hardwareconfig.h and feed it to function that will use it to update event information
        hardwaredata = myzip.read(r'custom/SetHardwareConfig.h').decode('utf-8')
        self.update_event_dict(hardwaredata)

        #decode SetStateMachineTP.h and feed it to function that will use it to update state information
        statedata = myzip.read(r'custom/SetStateMachineTP.h').decode('utf-8')
        self.update_state_dict(statedata)

        #update json file with new event and state values
        self.update_configfile()
     
    def update_state_dict(self, statedata):
        """Updates the names and numbers for the states part of configuration .json file using information found in a header file

        Parameters
        ----------
        statedata : str
            block of decoded text from the SetStateMachineTP.h file.
        """

        #turns block of text into list of strings where each string was seperated by a new line in the original file
        statedata = statedata.split('\n')
        #pull out the state machine list
        #starting index of the list is marked by #define at the beginning of the line and FOREACH_STATEMACHINE right before the list
        strt = str_in_list(statedata,['#define','FOREACH_STATEMACHINE'])
        #ending index is not clear, but I use the beginning of the next set of definitions to set an endpoint
        end = str_in_list(statedata[strt + 1:],'#define') + strt

        #list of states is all the definitions
        statelist = statedata[strt:end]

        #states are in order, use i to track number for state
        i=0
        #go through each line and pull out the name of the state for that number
        for val in statelist:
            #all state type definitions look like 'SM(name)'
            #this if statement picks out lines of that form and adds names and ids to my dictionary
            if 'SM(' in val:
                
                #take values between the paranthesis
                statename = take_inner(val,'(',')')

                #add event type to dictionary
                self.configfile['State_IDS'][str(i)] = statename
                i+=1
    
    def update_event_dict(self, hardwaredata):
        """Updates the names and numbers for the events part of configuration .json file using information found in a header file

        Parameters
        ----------
        hardwaredata : str
            block of decoded text from the sethardwareconfig.h file.
        """
        #turns block of text into list of strings where each string was seperated by a new line in the original file
        hardwaredata = hardwaredata.split('\n')
        #pull out the function sethardware config from .h file
        #starting index of the function is marked by SetHardwareConfig(){
        strt = str_in_list(hardwaredata,['SetHardwareConfig','()','{'])
        #ending index is  first '}' after the start
        end = str_in_list(hardwaredata[strt:],'}') + strt
        #pull out that function and assign it to it's own list 
        hrdwrcnfg = hardwaredata[strt:end]

        #update event ids in configfile.json using values pulled from the header file
        for val in hrdwrcnfg:
            #all event type definitions look like 'Create...(name,'name',id,....)'
            #this if statement picks out lines of that form and adds names and ids to my dictionary
            if 'Create' in val:
                #take values between the paranthesis
                eventlist = take_inner(val,'(',')')
                #remove whitespace
                eventlist = eventlist.replace(" ","")
                #turn string into list by splitting on commas
                eventlist = eventlist.split(',')

                #key is third item in paranthesis
                key = str(eventlist[2])
                #name is first item in paranthesis
                name = eventlist[0]
                #add event type to dictionary
                self.configfile['Event_IDS'][key] = name

    def new_dgroup(self):
        """Add a new group to the h5 file, used when a new session begins."""
        #check if any sessions have been stored yet
        if self.h5.keys():
            last_key = list(self.h5.keys())[-1]
            num = int(last_key.split('_')[-1])
        else:
            #if no sessions have been created then this is session 0
            num = 0
        newkey = 'sess_' + digitnum(num+1,self.max_figs)
        self.dgroup = self.h5.create_group(newkey)

    def close_dgroup(self):
        """Close the current dgroup, happens when an end session event is found. Group will be deleted if it does not contain taps."""
        
        if not 'Trials in Session' in self.dgroup.attrs:
            del self.h5[self.dgroup.name]
        else:
            #this changes the storage type of each column to be more memory efficient
            for col in self.dgroup.keys():
                temp = self.dgroup[col][0:self.dgroup.attrs['Trials in Session']]
                del self.dgroup[col]
                self.dgroup.create_dataset(col,data = temp)

    def add_attr(self,attr,val):
        """Add an attribute the the current dgroup, used to add information about a session.

        Parameters
        ----------
        attr : str
            name of the attribute in the h5 file
        val : any
            value to be stored for that attribute
        """
        self.dgroup.attrs[attr] = val
    
    def new_dsets_rows(self, n_rows = 100):
        """Add a set of rows to the tap data datasets, or create them if they don't yet exist.
        
        Parameters
        ----------
        n_rows : int, optional
            amount of rows to add when resizing. Defaults to 100.
        """
       
        #loop through each column
        for col in self.tap_columns.keys():
             #make sure the column has already been created
            if col in self.dgroup.keys():
                #check if the data is 1-dimensional
                if self.dgroup[col].ndim == 1:
                    newsize = self.dgroup[col].size + n_rows
                else:
                    newsize = self.dgroup[col].shape[0] + n_rows
                #add a row to the column
                self.dgroup[col].resize(newsize, axis=0)
            else:
                #if the column hasn't been created yet then create it
                self.dgroup.create_dataset(col, chunks = True, shape = (500,),maxshape = (None,),**self.tap_columns[col])

    def determine_filelist(self):
        """Determine the files in the directory that need to be parsed through. All files must start with the subject name. Files that have already been parsed are not included in the list."""
        self.filelist = []
        for path in os.listdir(self.datdir):
            #check if file is named with the subject first
            if path[0:len(self.configfile['Subject'])] == self.configfile['Subject']:
                #check if file has already been parsed
                if not path in self.configfile['Parsed_Files']:
                    self.filelist.append(path)
    
    def parse(self):
        """go through each file in the directory and extract information using the appropriate subroutines.
         Ignore files that have already been included. Update json file as needed.
        """
        #find out which files need to be parsed
        self.determine_filelist()
        #a number to keep track of how many files have been parsed
        n= 0
        #nmax is the number of files in the directory that have not already been parsed
        nmax = len(self.filelist)
        #loop through each dat file
        for path in self.filelist:

            #shows progress
            n+=1
            print(f'parsing file {n}/{nmax}',end = '\r')

            #exact path to the file
            datloc = self.datdir +  "\\" + path
            #if file is a .dat type
            if datloc.split('.')[-1] == 'dat':
                #update the active datfile
                self.active_datfile_path = datloc

                #loop through dataframe and pick out events corresponding to session information
                for i in self.active_datfile.index:
                    row = self.active_datfile.loc[i]
                    #event is start of a state machine that handles a tap session
                    if row['Event Type'] == 'SMStarted':
                        #add new row
                        self.new_dgroup()
                        #update start time and state name
                        self.add_attr('Start Time (Expt)',row['Time Since Experiment'])
                        self.add_attr('Session Type',row['State Name'])
                        self.add_attr('File',path)
                    #event reports upper limit for session
                    elif row['Event Type'] == 'TwoTapUpper':
                        self.add_attr('Upper',row['Value'])
                    #event reports lower limit for session
                    elif row['Event Type'] == 'TwoTapLower':
                        self.add_attr('Lower',row['Value'])
                    #event reports target for session
                    elif row['Event Type'] == 'TwoTapTarget':
                        self.add_attr('Target',row['Value'])
                    #event is end of session
                    elif row['Event Type'] == 'SMFinished':
                        self.add_attr('End Time (Expt)',row['Time Since Experiment'])
                        # #end last session, delete row if no taps occured within session
                        self.close_dgroup()
                    #check if a tap has occured
                    elif row['Event Type'] == 'TwoTapData':
                        #check if this was the first tap in a group of taps
                        if row['Value'] == 1:
                            #first we increment the amount of taps in this session by 1
                            #if no taps have been recorded yet then this attribute doesn't exist, we need to add it as to 0
                            if not 'Trials in Session' in self.dgroup.attrs:
                                self.add_attr('Trials in Session',0)
                                self.new_dsets_rows()

                            #add a tap to the session tap counter
                            n_trials = self.dgroup.attrs['Trials in Session'] + 1
                            self.add_attr('Trials in Session', n_trials)

                            #now we need to check if there is enough space in the tap info columns for more data
                            #if there is not enough space we need to do a resize and add some rows
                            if n_trials > self.dgroup['Tap 1'].shape[0]:
                                self.new_dsets_rows()

                            #add to the tap counter for this set of taps
                            self.dgroup['N Taps'][n_trials - 1] = self.dgroup['N Taps'][n_trials-1] + 1
                            #next we record a value for tap 1 on
                            #we are looking for an event 'LeverDI' with a value 1 (this signifies a lever press) that occured right before the 'TwoTapData' Event
                            t1_on = self.find_event(i, {'Event Type':'LeverDI','Value':1}, pos = 'before', col = 'Time Since State Machine', stop_if = [{'Event Type':'TwoTapData','Value':1}, {'Event Type': 'SMStarted'}])
                            t1_expt_on = self.find_event(i, {'Event Type':'LeverDI','Value':1}, pos = 'before', col = 'Time Since Experiment', stop_if = [{'Event Type':'TwoTapData','Value':1}, {'Event Type': 'SMStarted'}])

                            #next we record a value for tap 1 off
                            #we are looking for an event 'LeverDI' with a value 0 (this signifies a lever release) that occured right after the 'TwoTapData' Event
                            t1_off = self.find_event(i, {'Event Type':'LeverDI','Value':0}, pos = 'after', col = 'Time Since State Machine', stop_if = [{'Event Type':'TwoTapData','Value':1}, {'Event Type': 'SMFinished'}])
                            self.dgroup['Tap 1'][n_trials-1] = (t1_on,t1_off,t1_expt_on)

                        #check if this was the second tap in a group of taps
                        elif row['Value'] == 2:
                            #add to the tap counter for this set of taps
                            self.dgroup['N Taps'][n_trials-1] = self.dgroup['N Taps'][n_trials-1] + 1

                            #record a value for tap 2 on
                            t2_on = self.find_event(i, {'Event Type':'LeverDI','Value':1}, pos = 'before', col = 'Time Since State Machine', stop_if = [{'Event Type':'TwoTapData','Value':1}, {'Event Type': 'SMStarted'}])

                            #record a value for tap 2 off
                            t2_off = self.find_event(i, {'Event Type':'LeverDI','Value':0}, pos = 'after', col = 'Time Since State Machine', stop_if = [{'Event Type':'TwoTapData','Value':1}, {'Event Type': 'SMFinished'}])
                            

                            #record a value for interval
                            interval = t2_on - t1_on
                            #np.max is used here because interval must be >0 but if tap2 never happens than the interval will be a negative number because `t2_on` will be 0
                            self.dgroup['Tap 2'][n_trials-1] = (t2_on,t2_off)
                            self.dgroup['Interval'][n_trials-1] = np.max(interval,0)
                        
                        #check if this row is an extra tap, value can't be 6 because that value does not represent a tap
                        elif row['Value'] != 6:
                            #add to the tap counter for this set of taps
                            self.dgroup['N Taps'][n_trials-1] = self.dgroup['N Taps'][n_trials-1] + 1
                            
                            t_on = self.find_event(i, {'Event Type':'LeverDI','Value':1}, pos = 'before', col = 'Time Since State Machine', stop_if = [{'Event Type':'TwoTapData','Value':1}, {'Event Type': 'SMStarted'}])
                            t_off = self.find_event(i, {'Event Type':'LeverDI','Value':0}, pos = 'after', col = 'Time Since State Machine', stop_if = [{'Event Type':'TwoTapData','Value':1}, {'Event Type': 'SMFinished'}])
                            self.dgroup['Extra Taps'][n_trials-1] = list(self.dgroup['Extra Taps'][n_trials-1]) + [t_on,t_off]

                        #if the value is 6 then this event indicates a lick has occured
                        else:
                            lick_time = row['Time Since State Machine']
                            self.dgroup['Lick Time'][n_trials-1] = lick_time
                    #check if row indicates a reward is being delivered
                    elif row['Event Type'] == 'GiveReward_Int':
                        self.dgroup['Reward'][n_trials-1]  = (row['Value'],row['Time Since State Machine'])
                    #check if row indicates the tone turning on or off
                    elif row['Event Type'] == 'SoundSetting':
                        #tone is turning on
                        if row['Value'] == 1:
                            #record tone on to use it when tone off happens
                            tone_on = row['Time Since State Machine']
                        #tone is turning off
                        elif row['Value'] == 0:
                            self.dgroup['Tone Time'][n_trials-1] = (tone_on,row['Time Since State Machine'])

   
            #if file is a .zip type
            elif datloc.split('.')[-1] == 'zip':
                self.update_config_from_zip(datloc)

            self.configfile['Parsed_Files'].append(path)
            self.update_configfile()
        print(f'finishing up',end = '\r')
        self.finish_parsing()

    def finish_parsing(self):
        """This method runs after parsing is complete, it is used to find any problems with the hdf5 file and fix them"""

        #check if any sessions still have unlimited maxshape, this is a waste of memory
        #this issue arrises when a session end event is skipped
        for key in self.h5.keys():
            if not 'Trials in Session' in self.h5[key].attrs:
                del self.h5[key]
            else:
                for subkey in self.h5[key].keys():
                    if isinstance(self.h5[key][subkey], h5py.Dataset):
                        if isinstance(self.h5[key][subkey].maxshape[0],type(None)):
                            datalen = self.h5[key].attrs['Trials in Session']
                            temp = self.h5[key][subkey][0:datalen]
                            del self.h5[key][subkey]
                            self.h5[key].create_dataset(subkey,data = temp)
         
    def find_event(self, index, param, col = slice(None), pos = 'before', max_delta_t = 3e5, fail = 0, stop_if = []):
        """Find a row in the active datfile that matches certain conditions.

        Parameters
        ----------
        index : int
            index to find the row relative to.
        param : dict
            dictionary describing the values that each row needs to have in the event row.
        pos : str, optional
            You can specify 'before' to look for the first row before `index` that matches `param` or 'after' to look for the first row after `index` that matches `param`. 
            Default is 'before'.
        max_delta_t : int, optional
            Amount of time in ms to look through before giving up. Defaults to 3e5 (5 minutes).
        fail : any, optional
            What to return if the row cannot be found, defaults to 0
        col : str, int, optional
            The column of the row to return a value from, returns the whole row if none is specified
        stop_if : list, optional
            Returns `fail` if a row matches any of these sets of conditions. Must be a list of dictionaries, each dictionary represents a row. Defaults to a blank list.
        
        Returns
        -------
        out : any
            Returns the whole row as a pd.Series if nothing is specified for the column argument, returns the value in that specific column if column argument is specified.

        Notes
        -----
        For example "find_event(4,{'Event Type':'LeverDI','Value':1},'after')" will return the first row after row 4 that has 'LeverDI' for the 'Event Type' column and a 1 in the 'Value' column.
        """
        start_time = self.active_datfile.iloc[index]['Time Since Experiment']
        delta_t = 0
        #selects operation to perform after each row check, index should be increasing if 'after' is specified and vice versa. 'pos.lower()' makes all letters lowercase so input is not case sensitive.
        operation = {'before':(lambda x:x-1),'after':(lambda x:x+1)}[pos.lower()]
        #keep checking rows until max iterations are reached
        while delta_t<max_delta_t and 0<=index<len(self.active_datfile)-1:
            #increment or decrement index
            index = operation(index)
            #select row corresponding to index
            row = self.active_datfile.iloc[index]
            #check if row matches specified conditions
            if row_is(row,param):
                #if there is a match then return the row
                return row[col]
            #check if row matches any of the conditions specified in stop_if
            for stop_row in stop_if:
                if row_is(row,stop_row):
                    return fail
            delta_t = abs(row['Time Since Experiment'] - start_time)
        return fail

    @classmethod
    def parsedirectory(cls, active_dir, datdir):
        """Parses through a directory that has already been prepared by the `new_dir_setup` class. 
        Looks through all new .dat files in `datdir` and adds their information to the h5 file in `active_dir`.
        
        Parameters
        ----------
        datdir : str
            Path to the directory where the .dat files are stored
        active_dir : str
            Path to the directory that contains the config.json file and the h5 file."""

        #the try/finally structure is used to ensure that the h5 gets closed even if the parser runs into an error
        try:
            tempcls = cls(active_dir,datdir)
            tempcls.parse()

        finally:
            tempcls.h5.close()