This directory contains python objects that can be used to parse through a directory filled with the '.zip' and '.dat' files that get produced by the teensy.

Initial Setup
-------------
    1. Copy and paste this folder somewhere local on your computer.
    2. In parse_dats.bat on line 2 change the path to your anaconda executable
    3. In parse_dats.bat on line 3 change to your desired environment
        Make sure the environment has pandas and h5py installed

Usage
-----
    1. Double click on the 'parse_dats.bat' file within this directory.
    2. You will be asked wether or not you have a text file:
        -if yes you will be prompted to select the text file
        -if no you will be prompted to select the directory to parse through
    3. You will now be asked if you want to include trajectory data.
        -If you want to just do the basic trial data without trajectories input n and press enter.
        -If you want to include trajectories and those h5 files are ready input y and press enter. 
        You will then be prompted to select the location of the h5 files, select it. 
        You will be asked if you want to select another location, this is useful if left and right video files are in separate directories. 
        If this is the case input y and press enter and then choose the next directory. 
        Keep choosing directories this way until you are done and then input n and press enter.
    4. The parser will parse through the directory or list of directories, if this is the first time it may take a while (~10min)
        -if this is not the first time the parser will simply append the data from new sessions at the end of the existing h5 file

Output Format
-------------
    Once the parser finishes the directory will be populated with a 'parser' subfolder, inside there will be two files:
        Config.json : This file contains the data used by the parser, this data is pulled from the '.zip' files within the directory.
            The numbers for the state machines and events are stored under 'Event_IDS' and 'State_IDS'
            The names of all the files that have already been parsed are stored in 'Parsed_Files'.
            The name of the subject is stored as well.

        `subjectname`.h5 : This is the hdf5 file that contains all of the session data.
            The file metadata contains the subjects name and box number
            Each session will be a subfolder with session information as its metadata
            The columns are set up as follows:
                Extra Taps: A list of tap on, tap off for each extra tap in a trial
                Lick Time: The time that "lick" event is reported for each trial
                N Taps: The amount of taps within a trial (usually 2)
                Reward: The value for each reward (1-5) and when it occured
                Tap 1: Tap 1 on, Tap 1 off, and the experiment time of Tap 1 on
                Tap 2: Tap 2 on, Tap 2 off
                Interval: Time between tap 1 on and tap 2 on
                Tone Time: Tone on and off
                L_trajectories: Contains a table for each trial with the trajectory data from the left camera
                R_trajectories: Contains a table for each trial with the trajectory data from the right camera


Directory Formatting
--------------------
    The path to the directory is expected to be: Drive:\...\Results-[Box]-[Subject]\Master-[#]
    If using a textfile each line should point to a directory in the way described above.

Files in this Directory
-----------------------
    event_parser_hdf5.py : The class responsible for parsing through '.zip' and '.dat' files and storing the session and trial info in an h5 file.
    init_config.json : The initial config file used for each new directory.
        This includes the event ids that are defined in the excel file but not include in the hardwareconfig.h file within each zip.
    new_dir_setup.py : This class is responsible for adding a blank '.h5' file and a 'config.json' file to each new directory.
    parse_dats.bat : This batch file is used to actually run the parser.
    runner.py : This script is called on by parse_dats.bat, it prompts the user for inputs and runs the parser on directories as needed.
    utils.py : These are simple subfunctions that are needed by other classes and scripts within this directory.
    traj_parser.py : This is the class responsible for parsing through trajectory data and adding the info to the main h5 file for each rat.