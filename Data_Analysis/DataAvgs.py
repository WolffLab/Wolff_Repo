
from Wolff_Repo.Utils.simple import *
import pandas as pd
import scipy as sp
import numpy as np

class DataAvgs:
    """Class used to average the data between several different rats."""

    def __init__(self, holderlist):
        """Initializes class.
        
        Parameters
        ----------
        holderlist : iterable
            A list of dataholder objects
            
        Returns
        -------
        out : DataAvgs"""
        self.holderlist = holderlist
        self.nrats = len(holderlist)

    def _average_group(self, group):
        """Take a list of series and average them to output a single series. Lists do not need to be the same length.
        
        Parameters
        ----------
        group : list
            list of pd.Series to average.
        
        Returns
        -------
        out : list
            A one dimensional list that averages all the lists in `group`. 
            For example, the first element of the list will be the average of the first elements in all the lists."""


        return pd.concat(group).groupby(level = 0).mean()
 
    def averaged_col(self, col, target = None):
        """Return a column of averaged values from all rats.
        
        Parameters
        ----------
        col : str
            The name of the column you want averaged, must be one of the columns from the DataHolder classes you instantiated with
        target : int, optional
            Include this if you only want values from a specific target"""

        if not isinstance(target, type(None)):
            cols = [i.get_by_target(target, col = col) for i in self.holderlist]
        else:
            cols = [i[col] for i in self.holderlist]
        return self._average_group(cols)

    def TrialSuccess(self, error, avgwindow = 100, target = None):
        """Returns an array with the average percentage of trials within a moving window where the trial IPI was 
        +- error % away from the target IPI. First computes this average for each rat then averages over all rats.
        
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
        cols = [i.TrialSuccess(error, avgwindow, target) for i in self.holderlist]

        return self._average_group(cols)
    


class Last2000:

    def __init__(self, ratlist):

        targets = self._set_of_targets(ratlist) 
        self.targetframe = self._preprocess(ratlist, targets) 
        cutoffs = self._cutoffs(targets)

        self.df, [rIPI, rCV, rSR]  = self._Last2000(cutoffs, targets)

        self.IPIs = rIPI
        self.CVs = rCV
        self.SRs = rSR


    def _preprocess(self, ratlist, targets):

        targetframes = []
        dataframes = []
        # for each target, 
        for rat in ratlist:
            for target in targets:
            # for each of the rats, 
                # grab the entries that have a specific target
                data = rat.df.loc[rat.df["target"]==target]
                data = data.reset_index()
                # append that data to the dataframe 
                dataframes.append(data)
            # append the dataframe for the whole target group to the targetframe 
            targetframes.append(dataframes)
            # reset the dataframe for the next target group 
            dataframes = []
        return targetframes 


    def _set_of_targets(self, rats):
        targets = [] 
        for rat in rats:
            tar = (rat.set_of('target')).tolist()
            for item in tar: 
                targets.append(item)
        target = list(set(targets)) 
        (target).sort(reverse=True)
        return target


    def _cutoffs(self, targets): 
        cutoffs = []
        targetcutoff = [] 
        for t in range(len(targets)): 
            for i in range(len(self.targetframe)):
                cutoff = self.targetframe[i][t].shape[0]
                targetcutoff.append(cutoff)
            cut = np.min(targetcutoff) 
            cutoffs.append(cut) 
            targetcutoff = [] 
        return cutoffs 

    def _calculate_SR(self, frame, err = 20):
        # grab the percentage error for each trial 
        loss = (frame['loss']).to_numpy()
        # define upper and lower bounds
        upper = err/100
        lower = -err/100 

        #convert a bool array of whether or not losses are in between bounds to integer array 
        success = ((loss <= upper) & (lower <= loss)).astype(int)

        SR = np.mean(success) 
        
        return SR


    def _Last2000(self, cutoffs, targets):

        IPIs = []
        CVs = []
        SRs = []
        AAh = []

        rIPI = []
        rCV = []
        rSR = [] 

        for t in range(len(targets)): 
            for i in range(len(self.targetframe)):
                q = self.targetframe[i][t].iloc[cutoffs[t]-2000 : cutoffs[t]]
                interval = q['interval'].to_numpy()

                IPI = np.mean(interval) 
                IPIs.append(IPI)

                CV = np.std(interval)/np.mean(interval)
                CVs.append(CV) 

                SR = (self._calculate_SR(q, err = 20))*100
                SRs.append(SR) 
            
            IPI = np.mean(IPIs) 
            rIPI.append(IPIs)

            CV = np.mean(CVs) 
            rCV.append(CVs)

            SR = np.mean(SRs) 
            rSR.append(SRs)

            frame = {"IPI": IPI, "CV": CV, "SR" : SR}
            Ah = pd.DataFrame(data = frame, index = [f'{targets[t]}'])
            AAh.append(Ah) 
            
            IPI = []
            CV = [] 
            SR = []
            IPIs = []
            CVs = []
            SRs = []


        return pd.concat(AAh), [rIPI, rCV, rSR] 


class ChunkyMonkey:

    def __init__(self, ratlist, increment, error = 20):

        # find all of the targets involved
        targets = self._set_of_targets(ratlist) 
        # make the targetframe which is the rats grouped by target
        self.targetframe = self._preprocess(ratlist, targets) 
        # find the cutoffs 
        cutoffs = self._cutoffs(targets)
        # use the cutoffs
        self._Ensure_AllRat(cutoffs, targets) 

        self.intervals = self._Split_Interval_by(increment, targets) 
        self.success = self._Split_SR_by(increment, targets, err = error)
        self.vars = self._Split_CV_by(increment, targets) 


    def _set_of_targets(self, rats):
        # grab all of the targets that are within the data 
        targets = [] 
        # for each rat
        for rat in rats:
            # grab the list of targets 
            tar = list(set(rat.df['target'].to_numpy()))
            # append the data into a flat list
            for item in tar: 
                targets.append(item)
        # take the set so there's only one of each
        target = list(set(targets)) 
        # sort from largest to smallest 
        (target).sort(reverse=True)
        # return the targets 
        if len(target) == 3:
            target = target[1:]
        return target


    def _preprocess(self, ratlist, targets):
        
        targetframes = []
        dataframes = []
        # for each target, 
        for rat in ratlist:
            for target in targets:
            # for each of the rats, 
                # grab the entries that have a specific target
                data = rat.df.loc[rat.df["target"]==target]
                data = data.reset_index()

                dataframes.append(data)

            # append the dataframe for the whole target group to the targetframe 
            targetframes.append(dataframes)
            # reset the dataframe for the next target group 
            dataframes = []

        return targetframes


    def _cutoffs(self, targets): 
        # initiate blank arrays for the cutoffs 
        cutoffs = []
        targetcutoff = [] 
        # for each of the targets 
        for t in range(len(targets)): 
            # and each of the targetframes 
            for i in range(len(self.targetframe)):
                # find the length of each rat's data by target
                cutoff = self.targetframe[i][t].shape[0]
                # add it to the array
                targetcutoff.append(cutoff)
            # # find the minimum length for each target group
            cut = np.min(targetcutoff) 
            # add that to the cutoffs 
            cutoffs.append(cut) 
            # clear the targetcutoff array for the next target group
            targetcutoff = [] 
        return cutoffs 


    def _calculate_SR(self, frame, err = 20):
        # grab the percentage error for each trial 
        loss = (frame['loss']).to_numpy()
        # define upper and lower bounds
        upper = err/100
        lower = -err/100 

        #convert a bool array of whether or not losses are in between bounds to integer array 
        success = ((loss <= upper) & (lower <= loss)).astype(int)

        SR = np.mean(success) 
        
        return SR


    def _Ensure_AllRat(self, cutoffs, targets):

        for t in range(len(targets)): 
            for i in range(len(self.targetframe)):
                # for each of the targetframes, cut off the data so all of the rats are included. 
                q = self.targetframe[i][t].iloc[: cutoffs[t]]
                self.targetframe[i][t] = q.copy()
      
        
    def _Split_Interval_by(self, inc, targets): 

        newframe = []

        for t in range(len(targets)): 
            Frame = pd.DataFrame()
            for i in range(len(self.targetframe)):
                # pull the current rat
                data = self.targetframe[i][t]
                data = data['interval'].to_numpy()
                # find the decimal value of the length that is over the increment
                a = (len(data)/inc)-int(len(data)/inc)
                # find the number of not a numbers that need to be added on 
                add = int(inc - np.round(a*inc))
                # make an array for the not a numbers 
                nan = np.full(add, np.NaN)
                # append the arrays together 
                test = np.append(data, nan)
                # and reshape into the correct format (the -1 lets numpy chose the correct length)
                test2 = np.reshape(test, (-1, inc))
                # for each group in the rat, 
                avg = []
                for g in range(test2.shape[0]):
                    # take the NaN mean (otherwise last entry is NAN)
                    a = np.nanmean(test2[g])
                    # append it onto the avgs list
                    avg.append(a)
                    # get the length of the columns so you can insert the column at the right place. 

                    # insert that into the frame 
                Frame.insert(0, f'Rat_{i}_{targets[t]}', avg) 
            # once all the rats have been added in, take the average
            avg = []
            for row in range(Frame.shape[0]):
                # take the mean of each row 
                a = np.mean(Frame.iloc[row].to_numpy())
                # and add it to a list 
                avg.append(a) 
                # insert it to the frame
            Frame.insert(0, 'Mean', avg) 

            newframe.append(Frame) 
        
        return newframe 


    def _Split_SR_by(self, inc, targets, err = 20): 

        newframe = []

        for t in range(len(targets)): 
            Frame = pd.DataFrame()
            for i in range(len(self.targetframe)):
                # pull the current rat
                data = self.targetframe[i][t]
                data = data['loss'].to_numpy()
                # find the decimal value of the length that is over the increment
                a = (len(data)/inc)-int(len(data)/inc)
                # find the number of not a numbers that need to be added on 
                add = int(inc - np.round(a*inc))
                # make an array for the not a numbers 
                nan = np.full(add, np.NaN)
                # append the arrays together 
                test = np.append(data, nan)
                # and reshape into the correct format (the -1 lets numpy chose the correct length)
                test2 = np.reshape(test, (-1, inc))
                # for each group in the rat, 
                avg = []
                for g in range(test2.shape[0]):
                    # take the NaN mean (otherwise last entry is NAN
                    upper = err/100
                    lower = -err/100 

                    #convert a bool array of whether or not losses are in between bounds to integer array 
                    success = ((test2[g] <= upper) & (lower <= test2[g])).astype(int)
                    a = np.nanmean(success)*100
                    # append it onto the avgs list
                    avg.append(a)
                    # get the length of the columns so you can insert the column at the right place. 

                    # insert that into the frame 
                Frame.insert(0, f'Rat_{i}_{targets[t]}', avg) 
            # once all the rats have been added in, take the average
            avg = []
            for row in range(Frame.shape[0]):
                # take the mean of each row 
                a = np.mean(Frame.iloc[row].to_numpy())
                # and add it to a list 
                avg.append(a) 
                # insert it to the frame
            Frame.insert(0, 'Mean', avg) 

            newframe.append(Frame) 
        
        return newframe 


    def _Split_CV_by(self, inc, targets): 

        newframe = []

        for t in range(len(targets)): 
            Frame = pd.DataFrame()
            for i in range(len(self.targetframe)):
                # pull the current rat
                data = self.targetframe[i][t]

                data = data['interval'].to_numpy()
                # find the decimal value of the length that is over the increment 
                a = (len(data)/inc)-int(len(data)/inc)
                # find the number of not a numbers that need to be added on 
                add = int(inc - np.round(a*inc))
                # make an array for the not a numbers 
                nan = np.full(add, np.NaN)
                # append the arrays together 
                test = np.append(data, nan)
                # and reshape into the correct format (the -1 lets numpy chose the correct length)
                test2 = np.reshape(test, (-1, inc))
                # for each group in the rat, 
                avg = []
                for g in range(test2.shape[0]):
                    # take the NaN mean and standard deviation of array (otherwise last entry is NAN
                    stdev = np.nanstd(test2[g])
                    mean = np.nanmean(test2[g])  

                    a = stdev/mean
                    # append it onto the avgs list
                    avg.append(a)
                    # get the length of the columns so you can insert the column at the right place. 

                    # insert that into the frame 
                Frame.insert(0, f'Rat_{i}_{targets[t]}', avg) 
            # once all the rats have been added in, take the average
            avg = []
            for row in range(Frame.shape[0]):
                # take the mean of each row 
                a = np.mean(Frame.iloc[row].to_numpy())
                # and add it to a list 
                avg.append(a) 
                # insert it to the frame
            Frame.insert(0, 'Mean', avg) 

            newframe.append(Frame) 
        
        return newframe 


    def Find_CVDiff(self):
        """ Find the difference in coeficcient of variation between the two target IPIs. Based on chunk size from initiation of class. """
        # grab the data for the variations
        d1 = self.vars[0]
        d2 = self.vars[1]
        # cut where to cut the data for equal rows 
        if len(d1) > len(d2):
            cut = len(d2)-1
        else:
            cut = len(d1)-1
        # cut the data 
        cut1 = d1.iloc[:cut].copy()
        cut2 = d2.iloc[:cut].copy()

        for cut in [cut1, cut2]:
            i=len(cut.columns)-1
            for colname in cut.columns:
                if f'Rat_{i}_' in colname:
                    cut.rename(columns = {colname: i}, inplace = True)
                    
                i -= 1
            # find the difference by subtracting the whole dataframe
            
        dif = cut1 - cut2

        return dif 


    def Find_IPIDiff(self):
        # grab the data for the variations
        d1 = self.intervals[0]
        d2 = self.intervals[1]
        # cut where to cut the data for equal rows 
        if len(d1) > len(d2):
            cut = len(d2)-1
        else:
            cut = len(d1)-1
        # cut the data 
        cut1 = d1.iloc[:cut].copy()
        cut2 = d2.iloc[:cut].copy()

        for cut in [cut1, cut2]:
            i=len(cut.columns)-1
            for colname in cut.columns:
                if f'Rat_{i}_' in colname:
                    cut.rename(columns = {colname: i}, inplace = True)
                    
                i -= 1
            # find the difference by subtracting the whole dataframe
            
        dif = cut1 - cut2

        return dif 


    def Find_SRDiff(self):
        # grab the data for the variations
        d1 = self.success[0]
        d2 = self.success[1]
        # cut where to cut the data for equal rows 
        if len(d1) > len(d2):
            cut = len(d2)-1
        else:
            cut = len(d1)-1
        # cut the data 
        cut1 = d1.iloc[:cut].copy()
        cut2 = d2.iloc[:cut].copy()

        for cut in [cut1, cut2]:
            i=len(cut.columns)-1
            for colname in cut.columns:
                if f'Rat_{i}_' in colname:
                    cut.rename(columns = {colname: i}, inplace = True)
                    
                i -= 1
            # find the difference by subtracting the whole dataframe
            
        dif = cut1 - cut2

        return dif


class SessionAverages:


    def __init__(self, rats, target, columnname, numsessions = 10, window = 50, minwindow = 10, length = False): 
        """ Initialize the class. 
        
        Params 
        ---
        rats : list
        List of the rats within a group. 

        target : int
        Number of the target IPI
        
        length : int
        Integer value the sessions should be cut to.
        Default = False; length is calculated as the median length of all sessions. 
        """
        
        # if length was false, calculate the length. 
        if length != False:
            self.length = length 
        else: 
            self.length = self._calc_median_length(rats, target) 
        
        self.eq_sess = self._calc_isometric_sess(rats, target, columnname)

        self.X_len_sess = self._calc_by_session(numsessions)

        lenn = self.find_min_sess(self.X_len_sess)

        self.MainFrame = self._avg_n_smooth(self.X_len_sess, lenn, win = window, minwin = minwindow)

        
    def _calc_median_length(self, rats, target):
        """ Calculate the median length of all sessions within a rat group. 
        
        Params 
        ---
        rats : list
        List of the rats within a group. 

        target : int
        Value of the target IPI desired
        
        Returns 
        --- 
        length : int
        Integer value the sessions should be cut to.
        """
        # define blank arrays 
        sesslens = []
        sessframe = []
        # for each of the rats
        for rat in rats: 
            # pull the rats' info for the target specified
            ratt = rat.press_is(f'(target == {target})')
            # pull out the n_sess column and then sort so each n_sess value appears only once. 
            sessnums = list(set(ratt['n_sess'].to_numpy()))
            # add the things to sessframes (will be used later)
            sessframe.append(sessnums)

            # for all of the n_sess values
            for i in sessnums:
                # find the shape of the session. 
                leng = (ratt.loc[ratt['n_sess']==i]).shape[0]
                # append it 
                sesslens.append(leng)

        # self the sessframes so we can use them 
        self.sessframe = sessframe 

        # return the median value. 
        return np.median(sesslens)


    def _calc_isometric_sess(self, rats, target, columnname): 
        """ Find ??? 
        
        Params 
        ---
        rats : list
        List of the rats within a group. 

        target : int
        Value of the target IPI desired
        
        Returns 
        --- 
        Eq_SessFrame : list 
        Nested list of equal-length session dataframes for each rat. 
        """

        Eq_SessFrame = []

        # for each of the rats in the group,
        for i in range(len(rats)):
            # initialize blank array for isometric session data
            store = []
            # grab the sessions whose targets are 700
            ratt = rats[i].press_is(f'(target == {target})')
            # for each numerical value within the session list (nested list w/ seperate list per rat) 
            for j in self.sessframe[i]:
                # pull that session's data
                data = ratt.loc[ratt['n_sess']==j]
                # if the session has more than 50 taps, continue
                if len(data) > 50:
                    # Interpolate if the length of the session is less than the median
                    if len(data) < self.length:
                        ipi = data[f'{columnname}'].to_numpy()
                        f = sp.interpolate.interp1d(np.arange(len(ipi)), ipi, kind = 'slinear')
                        # we need interp_val datapoints but we're pulling from a smaller dataset.. 
                        # we need to sample at decimal point values between 0 and len(og data) 
                        space = (len(data)-2)/(self.length)
                        # define the array of sampling x values
                        newx = np.arange(0, len(data)-2, space)
                        # calculate the new ipi data and save it as a dataframe entry 
                        new = pd.DataFrame({f'{j}': f(newx)})
                        # append the data list onto the growing dataframe. 
                        store.append(new)

                    # Randomly sample if the length of the session is greater than the median. 
                    if len(data) >= self.length:
                        # take a copy of the interval
                        ipi = data[[f'{columnname}']].copy()
                        # randomly select N rows, where N is the median session length
                        randind = np.random.choice(np.arange(len(ipi)), int(self.length), replace = False)
                        # make a copy of the ipi dataframe with only the random id's pulled out 
                        new = ipi.iloc[randind].copy()
                        # sort the cropped dataframe by index value (meaing random taps are preserved temporally)
                        new.sort_index(axis=0, inplace = True)
                        # reset the index from N <-> N+length to 0 <-> length 
                        new.reset_index(drop=True, inplace = True)
                        # rename the column from 'interval' to the session #
                        new.rename(columns = {f'{columnname}': f'{j}'}, inplace = True)
                        # append the data list onto the growing dataframe. 
                        store.append(new) 
            
            # after everything has been added, concatenate the store matrix by column
            ah = pd.concat(store, axis =1)    
            Eq_SessFrame.append(ah)
        
        # return the full session frame. 
        return Eq_SessFrame 


    def _calc_by_session(self, numsessions): 
        """ Calculate the session groups from the equal-length frame"""

        # Initialize a mainframe 
        Frame = []
        # for each of the rats in the equal length frame, 
        for k in range(len(self.eq_sess)):
            # pull out the dataframe for the rat
            fra = self.eq_sess[k]
            # initialize the meanframe
            meanframe = []
            # for each group of X sessions to the length of the rat's sessions, 
            for i in np.arange(0, fra.shape[1], numsessions):
                # pull the data from the dataframe: all rows and columns i to i+numsessions (10 or 20 usually) 
                # then drop the last entry cause sometimes it's NaN. 
                data = fra.iloc[:, i:i+numsessions][:int(self.length -1)].copy()
                # initialize an average list
                avg = []
                # for each row in the data, 
                for row in range(data.shape[0]):
                    # take the mean of each row 
                    a = np.mean(data.iloc[row].to_numpy())
                    # and add it to a list 
                    avg.append(a) 
                # make a dataframe for the mean of those rows
                means = pd.DataFrame({f'Mean {int(i/10)}': avg}) 
                # reset the average list 
                avg = []
                # append the baby dataframe to the meanframe for concatenation later 
                meanframe.append(means) 
            
            # after everything has been added, concatenate the mean matrix by column
            MeanFrame = pd.concat(meanframe, axis = 1)
            # append the meanframe for each rat into the main frame. 
            Frame.append(MeanFrame) 

        return Frame 


    def find_min_sess(self, Frame):

        lengths = [] 
        for i in range(len(Frame)):
            lengths.append(Frame[i].shape[1])
        
        return np.min(lengths) 


    def _avg_n_smooth(self, Frame, lenn, win = 50, minwin = 10):
        """ Smooth the data"""
        # initialize a blank list for the means. 
        fra = []
        # for all of the columns in the rats' data 
        for k in range(lenn):
            # initialize a blank list for the means. 
            avg = [] 
            # for all of the rats in the mainframe, 
            for i in range(len(Frame)):
                # pull each rats' N-session average for that row 
                avg.append(Frame[i].iloc[:,k])
            # Concatenate the rats' N-session average for that row 
            new = pd.concat(avg)
            # Take the mean for that row. 
            mean = new.groupby(level=0).mean()
            # smooth 
            newdata = mean.rolling(win, min_periods=minwin).mean()
            # make it into a mini dataframe
            new = pd.DataFrame({f'Mean {i}': newdata}) 
            # append it into the blank list
            fra.append(new)
        # Concatenate all of the mini frames by column. 

        MainFrame = pd.concat(fra, axis = 1)
        return MainFrame 
    