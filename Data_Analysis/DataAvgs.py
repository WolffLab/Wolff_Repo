
from Wolff_Repo.Utils.simple import *
import pandas as pd
import scipy as sp


class DataAvgs:
    """ Class for inputting multiple rat press data files """


    def __init__(self, ratlist):
        """Initialization stores two dataframes in the class, one with information about specific presses 
        and one with the general information of each session. 

        Parameters
        --- 
        ratlist : list
            List of DataHolder objects. 
        Returns
        --- 
        None. Creates an object. 
        """

        datalist = [] 
        # pull the dataframes from the DataHolder object list
        for rat in ratlist:
            frame = rat.df 
            datalist.append(frame)

        self.ratlist = ratlist 

        # assume that all of the rats have the same list of targets 
        # just pull the first rat's targets
        targets = (ratlist[0].set_of('target')).tolist()
        # sort the targets from largest to smallest so 
        # they are always pulled in the same order. 
        (targets).sort(reverse=True) 
        self.targets = targets

        # do the preprocessing 
        targetlist = self._preprocessing(datalist)

        # get the list of cutoff values
        cutofflist = self._find_cutoff_values(targetlist)
        # make it into a self-referencing variable
        self.cuts = cutofflist

        # average the rats into a single dataframe per target
        targetframe = self._averaging(targetlist) 
        # make it into a self-referencing variable
        self.targetframe = targetframe 


    def _frames_by_target(self, datalist):
        """ Internal Function. Groups the dataframes by each of the targets contained. 
        Parameters 
        --- 
        datalist : list
            nested list of dataframes
        targets : list
            list of the targets used in the training 
        
        Returns 
        --- 
        targetlist : list 
            nested list of dataframes sorted by target 
        """
        # make empty lists for the things being seperated by targets
        targetframes = []
        dataframes = []
        # for each target, 
        for target in self.targets:
            # for each of the rats, 
            for frame in datalist:
                # grab the entries that have a specific target
                data = frame.loc[frame["target"]==target]
                data = data.reset_index()
                # append that data to the dataframe 
                dataframes.append(data)
            # append the dataframe for the whole target group to the targetframe 
            targetframes.append(dataframes)
            # reset the dataframe for the next target group 
            dataframes = []
        return targetframes 


    def __getitem__(self,key):
        """ Python Internal. 
        Indexing method of the class"""

        # Check that the key is an interger 
        if isinstance(key,int):
            # if yes, return that number press
            return self.presses.iloc[key]

        # Check that the key has two items
        elif isinstance(key,tuple):
            # key must be formatted as [number, column title]
            n,col = key
            # pull the nth press out 
            row = self.presses.iloc[n]

            # if the column title exists in the press data, 
            if col in self.presses:
                # return that column
                return row[col]

            # if the column title exists in the session data, ex  "target" or "upper" 
            elif col in self.sessions:
                # run get sess params function to return the column from session data 
                return self.get_sess_params(int(row.name[0]))[col]
        
        # Check that the key is only a string,
        elif isinstance(key,str):

            # if the column title exists in the press data, 
            if key in self.presses.columns:
                # return all rows from that column 
                return self.presses[key]

            # if the column title exists in the session data, 
            elif key in self.sessions:
                # return all rows from that column 
                return self.sessions[key]
            
            # if the column title is "n_sess" or "n_in_sess".
            elif key in self.presses.index.names:
                # return all rows from that column 
                return self.presses.index.get_level_values(key)
            
            else: # if the column title is none of the above, 
                raise Exception(f"{key} is not valid")

        else: # if the format of the key is none of the above
            raise Exception(f"{type(key)} is an invalid type")
    

    def _preprocessing(self, datalist):
        """ Internal Function. Crop the dataframes to include only specific columns & sort by 
        target value.
        Parameters 
        --- 
        datalist : list
            nested list of dataframes
        targets : list
            list of the targets used in the training 
        
        Returns 
        --- 
        targetlist : list 
            nested list of dataframes sorted by target 
        """

        # assume that all of the dataframes have the same targets witin each rat group
        # Seperate the data by target group 
        targetframes = self._frames_by_target(datalist)
        # create blank targetlist for storing the new dataframes and blank dataframes for iterating over 
        targetlist = []
        dataframes = []
        # for each of the target groups, 
        for target in targetframes:
            # for each of the dataframes within the target group, 
            for frame in target:
                # grab only the columns that we want to average over
                press = frame.copy()[["interval", "tap_1_len", "tap_2_len", "ratio", "loss", "upper", "lower"]]
                # append them onto the dataframe
                dataframes.append(press)
            # append each full dataframe to the target group dataframe
            targetlist.append(dataframes)
            # reset the dataframes list for the next target group
            dataframes = []
        return targetlist  


    def _find_cutoff_values(self, targetlist):
        """ Internal Function. Finds the 'cutoff' values -- where the avg will go from having N rats to N-1 rats. 

        Parameters 
        --- 
        targetlist : list
            nested list of dataframes seperated by target. 
        
        Returns 
        --- 
        cutofflist : list 
            nested list of lists of cutoff values for each target group 
        """
        # create blank arrays for the cutoff value list and a blank array for use in the for loops
        cutoff = [] 
        cutofflist = []
        # for each of the target groups, 
        for target in targetlist:
            # for each of the rats within the target groups, 
            for frame in target:
                # append the length of the dataframe to the cutoff list 
                cutoff.append(frame.shape[0])
            # append the list of cutoffs for each dataframe within the target group 
            cutoff.sort()
            cutofflist.append(cutoff)
            # reset the blank array for the next target group 
            cutoff = [] 
        # return the full cutoff list of all target groups. 
        return cutofflist  


    def _averaging(self, targetlist):
        """ Internal Function. Output will be a list of averaged dataframes, length of which will be the number of target ipi's used in training. 
        Parameters 
        --- 
        targetlist : list
            nested list of dataframes seperated by target. 
        
        Returns 
        --- 
        targetframe : list 
            nested list of averaged dataframes for each target group
        """ 
        
        # initiate the iteration variable at 0 
        i = 0
        # make a blank targetframe for the averaged dataframes. 
        targetframe = [] 
        # while there are still targetlists to go over, 
        while i < len(targetlist): 
            target = self.targets[i]
            # grab the datframes by the target group
            data = targetlist[i]         
            # concatenate the data
            newpress = pd.concat(data)
            # use builtin .groupby() to stack the frames and then average. 
            avg = newpress.groupby(level=0).mean()
            # If the number of columns in newpress is not equal to the # of columns in 'avg', then column(s) have been dropped. 
            if len(avg.columns) != len(newpress.columns):
                # pull the columns from the data
                col1 = data[0].columns.to_list()
                # pull the columns from the averaged data, 
                col2 = avg.columns.to_list()
                # find the differences between the two. 
                diff = list(set(col1).difference(col2))
                # pull the columns that have been dropped out of the newpress frame
                missing = data[0].copy()[diff]
                # append that column to the avg dataframe. 
                for i in range(len(diff)):
                    avg[f'{missing.columns[i]}'] = missing.iloc[:,i]
                # then continue... 

            # reset the indicies to be from 0 to N
            avg.reset_index()
            # automatically gets rid of the target column if we had it, so add that on for future reference 
            avg['target'] = np.full(avg.shape[0], target)
            # append the data entry to the single averaged dataframe. 
            targetframe.append(avg) 
            # step up i
            i += 1 
        # return the list of averaged dataframes. 
        return targetframe

    
    """ -------------------------------------------------------------------------------------------------- """

    def Find_TargetFrame(self, target):
        # have to find the targetframe that needs to be pulled 
        i=0 
        # while i is less than the number of targets,
        while i < len(self.targetframe):
            if (self.targetframe[i]).iloc[0,7] == target:
                break
            else: 
                i += 1
        
        return self.targetframe[i]


    def Find_i(self, target):
        # Find the i of the targetframe that needs to be pulled. 

        # make an empty list for the targets. 
        targs = [] 
        for i in range(len(self.targetframe)):
            # append all of the target values inside the frame to the targs list
            targs.append(self.targetframe[i]['target'][0])

        # if the target does not exist within this data, 
        if target not in targs:
            # return none if it's not
            return False
        # otherwise iterate thru the targetframes to find the i. 
        else:
        # have to find the targetframe that needs to be pulled 
            i=0 
            # while i is less than the number of targets,
            while i < len(self.targetframe):
                if (self.targetframe[i]['target'][0]) == target:
                    break
                else: 
                    i += 1
        return i
  
    
    def ShowOverview(self):
        """ Returns overview of the data. 
        
        Params 
        --- 
        None
        
        Returns
        info : print 
            Info about how many presses, targets, and rats included. 
        """
        numrat = "error" # len(self.cuts[0])
        targets = []
        for frame in self.targetframe:
            length = frame.shape[0]
            target = frame.loc[0,"target"]
            targets.append(target)
        text = f"Number of Rats: {numrat} ... Number of Trials: {length} ... Targets: {targets} "
        return text 
        

    def CutoffValues(self):
        return self.cuts 


    def MovingAverage(self, target, column, win = 1000, minwin = 1, boxcar = 300, err = 20):
        """Returns an array with the moving average of all trial data from the specified column/data. 
        
        Parameters 
        -------
        target : int 
            REQUIRED 
            The numerical value of the target desired. 

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

        frame = self.Find_TargetFrame(target)

        if column in frame.columns:
            data = frame[column]
            # use the pandas built-in 'rolling' to calculate the moving average. 
            # and assign the array output to 'avgs'
            avgs = (data.rolling(win, min_periods=minwin).mean())
            return avgs 
        
        if column == ("Success" or "success"):
            # grab the percentage error for each trial 
            loss = (frame['loss']).to_numpy()
            # define upper and lower bounds
            upper = err/100
            lower = -err/100 

            #convert a bool array of whether or not losses are in between bounds to integer array 
            success = ((loss <= upper) & (lower <= loss)).astype(int)
            # make the data into a dataframe
            df = pd.DataFrame(success, columns = ['Success'])
            # use the pandas built-in 'rolling' to calculate the moving average. 
            # and assign it to 'avgs'
            avgs = (df.rolling(win, min_periods=minwin).mean())*100
            # return the averages
            return avgs
        
        if column == ("CV" or "cv"):

            # grab the interval column from the prespecificied dataframe. 
            data = frame['interval']

            # find the average for the intervals. 
            avg = (data.rolling(win, min_periods = minwin).mean())
            # find the standard deviation for the intervals 
            sdev = (data.rolling(win, min_periods = minwin).std())
            # define the rough coefficient of variation as the standard deviation divided by the mean. 
            roughcv = sdev/avg
            # smooth the coefficient of variation with the moving average 
            cv = (roughcv.rolling(boxcar, min_periods=1).mean())
            # make sure any not a numbers are numpy not a number so they aren't plotted and don't break the matplotlib
            # then convert to numpy
            avgs = (cv.replace(pd.NA, np.NaN)).to_numpy()

            # return the dataframe as a numpy array 
            return avgs


    def DeltaIPI(self, target1, target2, norm, window = 2000, minwin = 100): 
        """Returns an array with the change in IPI's produced by a rat group. 
        
        Parameters 
        -------
        target1 : int 
            The numerical value of the first target desired. 

        target2 : int 
            The numerical value of the second target desired. 

        norm : str
            Whether or not you want the data normalized to one. 
        
        window : int
            Length of trials to the moving average. Default is 2000
        
        minwin : int 
            Length for the start of the moving average. Default is 100

        Returns 
        ------
        avgs : np.array
            Contains the change in interval
        """

        i = self.Find_i(target1)
        t1 = self.targetframe[i]['interval'].to_numpy()

        i = self.Find_i(target2)
        t2 = self.targetframe[i]['interval'].to_numpy()
        
        # cut the trials to the same length 
        if len(t1) > len(t2):
            cut = len(t2)
        elif len(t1) < len(t2):
            cut = len(t1)
        elif len(t1) == len(t2):
            cut = len(t1) 
        
        tt1 = t1[:cut]
        tt2 = t2[:cut] 
        
        # find difference of each tap, with absolute value if target2 < target1 
        d_ipi = abs((tt2-tt1))
        # divide by the supposed max distance if norm is true
        if norm != False:
            d_ipi = d_ipi/abs((target2-target1))
        # then take the moving average to smooth everything out. 
        d_ipi = pd.Series(d_ipi).rolling(window, min_periods=minwin).mean()

        # return it
        return d_ipi


    def CutByRat(self, dataframe, i, cv): 
        """ Returns an array with the coefficient of variation in each group by target IPI 
        
        Parameters 
        -------
        dataframe : dataframe 
            averaged dataframe for a specific target.

        i : int 
            The number in the targetframe that we are using (based on target wanted) 
            
        cv : np.array
            Contains the coefficient of variation for each tap

        Returns 
        ------
        ints : dict
            Dictionary of the cv data split up by the rat

        taps : dict
            Dictionary of the tap numbers split up by the rat         

        """

        # grab the list of cuts from the targetframe dataframe that was given as an argument
        cuts = dataframe.cuts[i]
        # find the number of cuts (ie. the number of rats in that frame) 
        leng = len(cuts)
        # initiate a dictionary
        ints = dict()
        # The first "cut" will be from the 0th trial to the first cut -- where all of the rats are averaged. 
        cutt = cv[:cuts[0]]
        # update the dictionary to include this value. 
        ints.update({'0': cutt})
        # for the number of rats in this frame, 
        for i in range(leng-1):
            # make another slice of the data from the ith to the ith+1 cut values 
            # each subsequent slice will be where there is one less rat being averaged over. 
            cutt = cv[cuts[i]:cuts[i+1]]
            # add that slice to the dictionary 
            ints.update({f'{i+1}': cutt}) 

        # initiate a dictionary for the taps
        taps = dict()
        # make a dictionary entry of a range of values from 0 to the first cut 
        # will be used for plotting so the 1st entry in the ints dictionary are the values for the 
        # taps in the 1st entry of the taps dictionary 
        taps.update({'0':range(cuts[0])}) 
        # for the number of rats, 
        for i in range(leng-1):
            # keep adding dictionary entries -- each one will be a range from XX to YY where there is a different 
            # number of rats being averaged over. These tap trial numbers match up to the ints dictionary
            taps.update({f'{i+1}':range(cuts[i], cuts[i+1])}) 
        
        # return both the ints and the taps for plotting. 
        return ints, taps 


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
    