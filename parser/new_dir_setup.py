import json, h5py, os

class new_dir_setup:
    """Setup a new directory for use with event parser. Creates a config file and blank HDF5 file. Determines name and box of subject from directory name."""
    #path to config file
    configpath = r'init_config.json'

    def __init__(self,dir):
        """Inititialize the new_dir_setup class

        Parameters
        ----------
        dir : str
            path to the directory that needs to be setup. Should be of the form 'Drive:\\path\\Results-boxname-ratname\\Master-#'
        
        Returns
        -------
        out : new_dir_setup
        """
        self.dir = dir
        self.get_subject_info()
        self.init_configfile()

    
    def init_configfile(self):
        """Uses the path 'configloc' taken from init to pull the json file, after this function is ran you can call configfile and use it like a dictionary"""
        with open(self.configpath,'r') as jsonfile:
            self.configfile = json.load(jsonfile)

    def get_subject_info(self):
        """Get the subject name and box name from the name of the directory"""
        filename = self.dir.split('\\')[-2]
        self.box,self.subject = filename.split('-')[1:]
    
    def dump_files(self):
        """Create a subdirectory inside the directory called 'parser' with a blank h5 file and a config.json file. These files will be used by the parser class later."""
        parser_dir = self.dir + r'\parser'
        os.mkdir(parser_dir)
        #store the name of the rat in the json file
        self.configfile['Subject'] = self.subject
        with open(parser_dir + r'\config.json','w') as jsonfile:
            #create the jsonfile in the parser directory
            json.dump(self.configfile, jsonfile, indent = 4)

        h5 = h5py.File(parser_dir + fr'\{self.subject}.h5','w')
        h5.attrs['Subject'] = self.subject
        h5.attrs['Box'] = self.box

    @classmethod
    def setup(cls,dir):
        """Add a parser directory to the directory with a blank h5 file and a config.json file. Determine box and rat name from directory name and add it to the information."""
        tempclass = cls(dir)
        tempclass.dump_files()