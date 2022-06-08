
from Wolff_Repo.Utils.simple import *
import pandas as pd
import h5py as h5

class DataHolder:
    """ Class for holding the rat press data files """


    def __init__(self, presses = "get", sessions = "get", dropafter = 0, dropfirst = 0, h5 = None, clean = True):
        """ Initialization takes two csv files, one with press informaion and one with session information. 
        A single dataframe is created that stores all the information about each press.
        If there is a drop after argument, all of the sessions after that number will be dropped 
        from the output array.  
    
        Parameters
        ----------
        presses : str, optional
            string containing the directory of the csv with press info, by default will open a ui to select file.
        sessions : str, optional
            string containing the directory of the csv with session info, by default will open a ui to select file.
        dropafter : int, optional
            defaults to 0, this won't drop anything.
        dropfirst : int, optional
            drop this number of trials from the beginning of the dataframe
        h5 : str, optional
            the path to an h5 file created by the parser class, if this value is included then presses and sessions are ignored.
        clean : bool, optional
            Wether to remove trials without a second tap, defaults to true.

        Returns
        -------
        out : Dataholder
            instance of DataHolder class, essentially a dataframe
        """

        if isinstance(h5, type(None)):

            # If the initialization of the class is left blank, 
            # open a file dialog to make the user chose the press info file. 
            if presses == "get":
                self.press_dir = get_file()
            # If the user already indicated a file, use it.
            else:
                self.press_dir = presses

            # If the initialization of the class is left blank, 
            # open a file dialog to make the user chose the press info file.
            if sessions == "get":
                self.sess_dir = get_file()
            # If the user already indicated a file, use it.
            else:
                self.sess_dir = sessions

            # do preprocessing of the dataframes. 
            self._init_df(dropafter)

        else:
            self.df = self._df_from_h5(h5)
            self._optimize_dtypes()

        # drop beginning trials if nececary 
        self.df = self.df.copy().iloc[dropfirst:,:]

        if clean:
            self.clean()


    def __getitem__(self, key):
        """ Python Internal. 
        Indexing method of the class"""

        if isinstance(key,str):
            return self.df[key]
        else:
            return self.df.loc[key]
    
    def _project_cols(self, df1, df2, projection_col, omit_col = None):
        """ add a column in df1 containing values from rows in df2 that share the same value of the argument "projection_col".
        
            Parameters
            ----------
            df1 : DataFrame
                DataFrame to project onto
            df2 : DataFrame
                Dataframe to copy values from
            projection_col : str
                column to use for matching the dataframes, column must exist in both dataframes
            omit_col : str, list, optional
                column or list of columns to ignore from df2, defaults to None. 

            Returns
            -------
            outdf : DataFrame
                copy of df1 with column added for each column in df2
            """

        outdf = df1.copy()
        copy_cols = list(df2.columns)
        if not isinstance(omit_col,type(None)):
            for col in omit_col:
                copy_cols.remove(col)
        
        for col in copy_cols:
            outdf[col] = pd.NA
            for n in df2.index:
                row = df2.loc[n]
                outdf.loc[df1[projection_col] == row[projection_col],col] = row[col]
                
        return outdf

    def _init_df(self, drop):
        """
        Initialize the dataframe with presses and sessions, uses the values for self.press_dir and self.sess_dir to make dataframes.

        Parameters
        ----------
        drop : int,optional
            drops sessions passed this number, ignores if drop = 0."""
        presses = pd.read_csv(self.press_dir)
        sessions = pd.read_csv(self.sess_dir)
        # self.sess_cols = sessions.drop(['starttime','sess_size'],axis=1).columns
        self.press_cols = presses.columns
        if not drop == 0:
            presses = presses.loc[presses['n_sess'] <= drop]
            sessions = sessions.loc[sessions['n_sess'] <= drop]
        self.df = self._project_cols(presses,sessions,'n_sess',['starttime','sess_size'])
        self._optimize_dtypes()

    def _sess_start_indices(self, presslist):
        """ return index of first press in each session within a list of presses """
        sesslist = np.sort(list(set(presslist['n_sess'])))
        indexlist = [0]
        for sess in sesslist[1:]:
            indexlist.append(indexlist[-1] + len(presslist.loc[presslist['n_sess'] == sess]))
        return (sesslist,indexlist)

    def _optimize_dtypes(self):
        """
        change each column in the dataframe to a more memory efficient datatype, columns and datatypes are specified in typedict.

        Parameters
        ----------
        None
        """

        typedict = {
            'reward' : 'category',
            'n_sess' : int,
            'n_in_sess' : int,
            'interval' : float,
            'tap_1_len' : float,
            'tap_2_len' : float,
            'ratio' : float,
            'loss' : float,
            'target' : 'category',
            'upper' : 'category',
            'lower' : 'category',
            'next_target' : 'category',
            'prev_target' : 'category'
        }

        if 'time' in self.df.columns:
            self.df['time'] = pd.to_datetime(self.df['time'])
        for key,val in typedict.items():
            if key in self.df.columns:
                self.df[key] = self.df[key].astype(val)

    def _df_from_h5(self, loc):
        """Return a df similiar to those used by DataHolder from the type of h5 file created by parser.
        
        Parameters
        ----------
        loc : str
            Path to h5 file.
    
        Returns
        -------
        out : pd.DataFrame
            Datafrom from h5 file."""

        out_dict = {i:[] for i in ['tap_1_len','tap_2_len','reward','n_sess','n_in_sess','interval','ratio','loss','target','upper','lower']}
        with h5.File(loc, 'r') as f:
            for dset in f:
                t_1_len = f[dset]['Tap 1']['Off'] - f[dset]['Tap 1']['On']
                interval = f[dset]['Interval'][:].astype(float)
                trials = f[dset].attrs['Trials in Session']
                target = f[dset].attrs['Target']
                upper = f[dset].attrs['Upper']
                lower = f[dset].attrs['Lower']
                out_dict['tap_1_len'] = out_dict['tap_1_len'] + list(t_1_len)
                out_dict['tap_2_len'] = out_dict['tap_2_len'] + list(f[dset]['Tap 2']['Off'] - f[dset]['Tap 2']['On'])
                out_dict['reward'] = out_dict['reward'] + list(f[dset]['Reward']['Value'])
                out_dict['n_sess'] = out_dict['n_sess'] + [int(dset.split('_')[-1])] * trials
                out_dict['n_in_sess'] = out_dict['n_in_sess'] + list(range(1,trials+1))
                out_dict['interval'] = out_dict['interval'] + list(interval)
                out_dict['ratio'] = out_dict['ratio'] + list(t_1_len/interval)
                out_dict['loss'] = out_dict['loss'] + list((interval - target)/target)
                for key in ['target','upper','lower']:
                    out_dict[key] = out_dict[key] + [eval(key)] * trials

        return pd.DataFrame(out_dict)
    
    @property
    def columns(self):
        return self.df.columns

    @property
    def sess_cols(self):
        """Return a list of columns that pertain to session values."""
        return [i for i in self.columns if i in ['target','upper','lower','next_target','prev_target','n_sess']]

    def clean(self):
        """Removes all trials where there is no tap 2 recorded."""
        self.df = self.df.loc[self.df['tap_2_len']!=0]
        self.df.index = range(len(self.df))

    def get_by_target(self, target, col = slice(None)):
        """ Returns all of the presses that have a particular target. 
    
        Parameters
        ----------
        target : int
            integer value of the interpress interval target
        col : str, optional
            string of specific column desired, default includes all columns. 

        Returns
        -------
        out : dataframe or series
            dataframe containing presses that have the desired target and desired columns.
        """
        return self.df.loc[self.df['target'] == target][col]

    def set_of(self, col, sort = True):
        """ Returns list of all data within specified column without duplicates. 
    
        Parameters
        ----------
        col : str
            string of specific column info desired. 

        sort : bool, optional
            wether or not to sort output, defaults to true.
        
        Returns
        -------
        out : list
        """

        if sort:
            return np.sort(list(set(self[col])))

        else: 
            return list(set(self[col]))

    def press_is(self, conditional_string):
        """ Returns numbered list of all presses whose columns meet particular values
    
        Parameters
        ----------
        conditional_string : str
            Conditional string describing parameters. 
            Column name options limited to columns in df

        Returns
        -------
        out : dataframe

        Examples
        --------
        DataHolder.press_is("target > 700") or DataHolder.sess_is("700 < target")
            returns dataframe with all presses whose target value is greater than 700 
        DataHolder.press_is("(target >= 500) & (interval > 10)")
            returns dataframe with all presses whose target is greater than or equal to 500 with an interval larger than 10
        """
        
        # for all of the columns in the dataframe,
        for col in self.df.columns:
            # check if the conditional string includes that column name. 
            if col in conditional_string:
                # if so, change the input conditional string to one that pandas can read
                # pandas needs dataframe.loc[dataframe['column']>x]
                conditional_string = sreplace(conditional_string,col,f"self.df['{col}']",count_as_alpha=['_'])

        # use pandas to apply formated conditional string and extract presses
        return self.df.loc[eval(conditional_string)]

    def get_first_press(self, conditional_string):
        """ Get the row of the first press that matches specific criteria
        
        Parameters 
        ----
        conditional_string : str
            Conditional string describing parameters. 
            Column name options limited to columns in df
        Returns
        ---
        outval : series
        """
        return self.press_is(conditional_string).iloc[0]

    def get_sess_params(self, n, col = slice(None)):
        """ Get the paramaters for a specific session
        
        Parameters 
        ----
        n : int
            number of session
        
        col : str, optional
            specific column if desired

        Returns
        ---
        outval : series
            row for the requested session
        """
        return self.get_first_press(f"n_sess == {n}")[self.sess_cols][col]

    def all_sessions(self, col):
        """ get the value of a parameter for all sessions
        
        Parameters 
        ----
        col : str, optional
            specific column you want the value for

        Returns
        ---
        outval : list
            the value of the column in every session
        """
        return [self.get_sess_params(i)[col] for i in self.set_of('n_sess')]

    def get_sess(self, n_sess):
        """ Get all presses within a particular session
        
        Parameters 
        ----
        n_sess : int
            number of session

        Returns
        ---
        outval : dataframe
            all presses within the requested session
        """
        return self.df[self.df['n_sess'] == n_sess]

    def change_target(self, old, new, save = False):
        """ change the target interval wherever it appears
        
        Parameters 
        ----
        old : int
            value of interval to be changed

        new : int
            value of new interval
        
        save : boolean
            wether or not to permenantly change csv. Default is false.

        Returns
        ---
        none
        """
        for n in self.df.index:
            if self.df.loc[n,'target'] == old:
                self.df.loc[n,'target'] = new

        if save:
            self.overwrite_sess()
    
    def drop_sess(self, n, save = False):
        """ delete a session
        
        Parameters 
        ----
        n : int or list
            number or list of numbers of sessions to be deleted
        
        save : boolean
            wether or not to permenantly change csv. Default is false.

        Returns
        ---
        none
        """

        #if integer, insert it into a list, this allows interger or list as input
        if isinstance(n,int):
            n = [n]
        
        #iterate through list of n values
        for i in n:
            self.df = self.df[~(self.df['n_sess'].isin(n))]

        if save:
            self.overwrite_press()
            self.overwrite_sess()

    def stats(self, stat, column, save = False):
        """ Compute a column of session statistics about a column from presses
    
        Parameters
        ----------
        stat : str
            statistic you want taken, can be mean,median,mode,max,min, or std

        column : str
            name of column from presses

        save: Boolean, optional
            wether or not to overwrite the csv

        Returns
        -------
        None

        """

        statcol = []
        for i in self.set_of('n_sess'):
            try:
                row = self.get_sess(i)[column].to_numpy()
                statcol.append(eval(f"row.{stat}()"))
            except KeyError:
                statcol.append(pd.NA)

        if save:
            self.overwrite_sess(**{column + "_" + stat : statcol})

        return statcol

    def TrialSuccess(self, error, avgwindow = 100, target = None):
        """ Returns an array with the percentage of trials within a moving window where the trial IPI was 
        +- error % away from the target IPI. 
        
        Parameters 
        -------
        error : int
            The numerical value of the percentage bounds from target desired. 
        avgwindow : int
            The number of sessions that should be used to calculate the moving average. 
            Default is a window of 100
        target : int, optional
            include this if you want to only take values from a particular target IPI
        
        Returns 
        ------
        avg : np.array
            Contains the moving average of the successes

        """
 
        # grab the percentage error for each trial
        if isinstance(target,type(None)):
            loss = (self.df['loss']).to_numpy()
        else:
            loss = self.get_by_target(target, col = 'loss').to_numpy()
        
        # define upper and lower bounds
        upper = error/100
        lower = -error/100 

        #convert a bool array of wether or not losses are in between bounds to integer array 
        success = ((loss <= upper) & (lower <= loss)).astype(int)
        # make the data into a dataframe
        df = pd.DataFrame(success, columns = ['Success'])
        # use the pandas built-in 'rolling' to calculate the moving average. 
        # and assign it to 'avgs'
        avgs = (df.rolling(avgwindow, min_periods=1).mean())*100
        # return the averages
        return avgs

    def MovingAverage(self, columnname, win = 100, minwin = 10, err = 20, box = 300):
        """ Returns an array with the moving average of all trial data from the specified column/data. 
        
        Parameters 
        -------
        columnname : str
            REQUIRED
            String of the column desired. Can be any of the columns in the dataframe, or 
            'success' and 'cv' 

        win : int
            OPTIONAL
            The number of sessions that should be used to calculate the moving average. 
            Default is a window of 100 
        
        minwin : int
            OPTIONAL
            The number of sessions used to calculate a single average. Ex. minwin = 20 will 
            Wait until the 20th row before calculating an average and place NaN in the first 19 rows.
            Default is a window of 10

        err : int
            OPTIONAL
            Whole number for the percentage bounds for the 'success' moving average. 
            Default is 20 ( +-20% bounds ) 
        
        boxcar : int
            OPTIONAL 
            Whole number for the coefficient of variation smoothing average. 
            Default is a window of 300
        
        Returns 
        ------
        avgs : np.array
            Contains the moving average of the desired column
        """

        if columnname in self.df.columns: 
            # pull the data out of the main dataframe
            data = self.df[columnname]
            # take the rolling average
            avgs = (data.rolling(win, min_periods=minwin).mean())
            return avgs 

        if columnname == 'success':
            # grab the percentage error for each trial 
            loss = (self.df['loss']).to_numpy()
            # define upper and lower bounds
            upper = err/100
            lower = -err/100 

            #convert a bool array of wether or not losses are in between bounds to integer array 
            success = ((loss <= upper) & (lower <= loss)).astype(int)
            # make the data into a dataframe
            df = pd.DataFrame(success, columns = ['Success'])
            # use the pandas built-in 'rolling' to calculate the moving average. 
            # and assign it to 'avgs'
            avgs = (df.rolling(win, min_periods=minwin).mean())*100
            # return the averages
            return avgs

        if columnname == ("CV" or "cv"):

            # grab the interval column from the prespecificied dataframe. 
            data = self.df['interval']

            # find the average for the intervals. 
            avg = (data.rolling(win, min_periods = minwin).mean())
            # find the standard deviation for the intervals 
            sdev = (data.rolling(win, min_periods = minwin).std())
            # define the rough coefficient of variation as the standard deviation divided by the mean. 
            roughcv = sdev/avg
            # smooth the coefficient of variation with the moving average 
            cv = (roughcv.rolling(box, min_periods=1).mean())
            # make sure any not a numbers are numpy not a number so they aren't plotted and don't break the matplotlib
            # then convert to numpy
            avgs = (cv.replace(pd.NA, np.NaN)).to_numpy()

            # return the dataframe as a numpy array 
            return avgs

    def SessionSuccess(self, error, avgwindow = 5):
        """ Returns an array with the percentage of successes in each session where the trial IPI was 
        +- error % away from the target IPI. 
        
        Parameters 
        -------
        error : int
            The numerical value of the percentage bounds from target desired. 
        avgwindow : int
            The number of sessions that should be used to calculate the moving average. 
            Default is a window of 5 
        
        Returns 
        ------
        successes : np.array
            Contains the percentage of succcesses within each session
        avg : np.array
            Contains the moving average of the successes per session

        """
        # create blank array for the session successes. 
        success = [] 
        # iterate through all of the sessions whose number of trials isn't zero.
        for i in self.set_of('n_sess'):
            # error in the csv is in decimal form, so convert to upper and lower bounds. 
            upper = error/100 
            lower = -error/100
            # pull all presses for a particular session
            data = self.get_sess(i)['loss']
            # perc_within returns percentage of values between lower and upper within data
            data = perc_within(data,lower,upper) * 100
            # append the number of successes for that session
            success.append(data)
        # create a new dataframe with the successes
        df = pd.DataFrame(success, columns = ['Success'])
        # use the pandas built-in 'rolling' to calculate the moving average. 
        # and add a column to the dataframe with the moving averages. 
        movingavg = df.rolling(avgwindow, min_periods=1).mean()
        avgs = movingavg.to_numpy()
        # return the two numpy lists.
        return success, avgs

    def overwrite_sess(self,**kwargs):
        """ overwrite the session csv using the location given when the class was instantiated
        
        Parameters 
        ----
        none

        Returns
        ---
        none
        """
        sessions = pd.read_csv(self.sess_dir)
        sessions = sessions.loc[sessions['n_sess'].isin(self.set_of('n_sess'))]
        sessions.sort(['n_sess'],inplace = True)
        for col in self.sess_cols:
            sessions[col] = self.df[col]

        if kwargs:
            for key,val in kwargs.items():
                sessions[key] = val
        sessions.to_csv(self.sess_dir)

    def overwrite_press(self):
        """ overwrite the presses csv using the location given when the class was instantiated
        
        Parameters 
        ----
        none

        Returns
        ---
        none
        """
        presses = pd.read_csv(self.press_dir)
        presses = presses.loc[presses['n_sess'].isin(self.set_of('n_sess'))]
        presses.sort_values(['n_sess','n_in_sess'],inplace=True)
        for col in self.press_cols:
            presses[col] = self.df[col]
        self.presses.to_csv(self.press_dir)