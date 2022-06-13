# 21 January 2022 
# Maegan Jennings, Ozzy Weinreb 

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy as sc
import datetime
from scipy import stats
from Wolff_Repo.Data_Analysis.DataHolder import DataHolder
from Wolff_Repo.Data_Analysis.DataAvgs import DataAvgs
from Wolff_Repo.Utils.simple import *
import math as ma 


class AveragedRats:

    farben = ['xkcd:slate grey', 'xkcd:deep green', 'xkcd:greyish green', 'xkcd:cool grey', 'xkcd:slate grey', 'xkcd:deep green', 'xkcd:greyish green', 'xkcd:cool grey']
    colors = ['xkcd:wine red', 'xkcd:grape', 'xkcd:dark lavender', 'xkcd:blueberry', 'xkcd:ocean blue', 'xkcd:turquoise', 'xkcd:light teal', 'xkcd:sage green', 'xkcd:yellow ochre', 'xkcd:pumpkin', 'xkcd:burnt umber']

    
    def __init__(self, ratdata, ratname):
        """ Class for plotting the rat data. This works on a single rat group's data, or a few rat groups' data. 
        Interacts with the DataAvgs objects. 
        
        Parameters 
        ---- 
        ratdata : object or list
            Name(s) of the object returned from the DataAvgs class
        ratname : string or list
            Name(s) of the rat to be displayed on the plots. 
        """

        if isinstance(ratname, str):
            self.names = [ratname]
        elif isinstance(ratname, list):
            self.names = ratname
        else:
            raise Exception(f"{type(ratname)} is invalid type for ratname, must be string or list.")

        if isinstance(ratdata, DataAvgs):
            self.rat = [ratdata]
        elif isinstance(ratdata, list):
            self.rat = ratdata
        else:
            raise Exception(f"{type(ratdata)} is invalid type for ratdata, must be list or DataAvgs.")
    
    def _xlabel(self, value, tick_number=None):
        if abs(value) < 1000:
            num = 0 
        else:
            num = ma.floor(ma.log10(abs(value))/3)
        value = round(value / 1000**num, 2)
        return f'{value:g}'+' KM'[num]
   
    def Moving_Average(self, data, window = 1001):
        """Return a moving average over `data`.
        
        Parameters
        ----------
        data : iterable
            data to average over
        
        window : int, optional
            size of the moving window, must be an odd number. 
            Defaults to 1001.
        
        Returns
        -------
        out : iterable
            Moving average, will be of the same type as the input iterable.
            
        Notes
        -----
        The length of the output structure will be the length of the input structure minus the windowsize."""

        return func_convolve(data, np.mean, window)

    def Moving_CV(self, data, window = 1001):
        """Return a moving coefficient of variation over `data`.
        
        Parameters
        ----------
        data : iterable
            data to to take cv over
        
        window : int, optional
            size of the moving window, must be an odd number. 
            Defaults to 1001.
        
        Returns
        -------
        out : iterable
            Moving cv, will be of the same type as the input iterable.
            
        Notes
        -----
        The length of the output structure will be the length of the input structure minus the windowsize."""

        return func_convolve(data, stats.variation, window)

    def Boxcar(self, data, window):
        """Return a boxcar smoothed iterable.
        
        Parameters
        ----------
        data : iterable
            data to perform boxcar on.
            
        window : int
            size of each boxcar
            
        Returns
        -------
        out : iterable
            boxcar smoothed data, same dtype as input iterable."""

        return boxcar(data, window)

    def Plot(self, ptype = 'IPI', target = 700, window = None, error = 10, boxcar = None, savepath = None, ylo = 0, yhi = "Default", target2 = 500, norm = True):
        """ Returns the requested plot with the requested parameters. 
        
        Parameters 
        ---------- 
        ptype : str
            REQUIRED
            Name of the plot that you want to make. 
            allowed strings: "IPI", "Success", "Tap1", "Tap2"

        target : int 
            REQUIRED
            Number value of the target IPI desired 
            Default is 700 
        
        window : int or None, Optional
            size of the moving average window if you want to take a moving average, defaults to None.

        minwindow : int 
            REQUIRED
            Number for the minimum window for moving average calculatons 
            Default is 100
        
        error : int
            Only required for Success plots
            Number for the error margins for the Success plots. 
            Default is 10% 

        boxcar : int
            Only required for the coefficient of variation plots
            Number for the window length for the moving average "smoothing" function
            Default is 300
        
        Returns 
        ---- 
        unnamed : matplotlib plot 
        """
        
        
        # Graph of Tap1 & IPI vs. trials 
        if ptype.lower() == 'tap1':
            self.Tap_vIPI(target, window = window)

        # Graph of Tap2 & IPI vs. trials 
        elif ptype.lower() == 'tap2':
            self.Tap_vIPI(target = target, tap = 2, window = window)

        # Graph of IPI vs. trials 
        elif ptype.lower() in ('ipi', 'interval'):
            self.IPI(target = target, window = window, ylo = ylo, yhi = yhi, save = savepath)

        # Graph of Success vs. trials
        elif ptype.lower() == 'success':
            self.Success(target = target, error = error, window = window, save = savepath)

        elif ptype.lower() == "cv":
            self.CV(target = target, window = window, box = boxcar, ylo = ylo, save = savepath)

        elif ptype.lower() in ("deltaipi", "delta ipi", "delipi", "del ipi"):
            print('plotting del ipi')
            self.DelIPI(target1 = target, target2 = target2, window = window, save = savepath, norm = norm)



    def DelIPI(self, target1, target2, win = None, save = None, norm = True): 
        """Plots the difference in average IPIs between two targets among the group of rats. 
        See `DataAvgs.DelIPI` for more info on data computation.
        
        Parameters
        ----------
        target1 : int
            Value of first target

        target2 : int
            Value of second target

        win : int or None, optional
            Size of window for moving average if you want one, must be an odd number.
            Defaults to None.

        save : str, optional
            A path to save the figure in, defaults to none.

        norm : bool, optional
            Wether or not to normalize using the difference of the two target IPIs, defaults to True."""
        
        # make a list of numbers to iterate over for the colors. 
        cols = np.arange(len(self.rat))

        # define the plotting style
        plt.style.use('default') 

        # now that all posibilities of self.rat are lists, 
        for rat, name, c in zip(self.rat, self.names, cols):
            delipi = rat.DelIPI(target1, target2, norm)
            if not isinstance(win, type(None)):
                delipi = self.Moving_Average(delipi, win)
            trials = range(len(delipi))

            plt.plot(trials, delipi, color = self.colors[c], label = f'{name}')
        
        #if norm != False: 
            #plt.ylim((0,1)) 
        #else: 
            #plt.ylim((-50,target2-target1))
        
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=3).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')
        plt.xlabel("Tap Number")
        plt.ylabel(r'$\Delta$IPI (ms)') 
        plt.title(r'$\Delta$IPI' + f' between {target1}ms and {target2}ms Trials') 

        if not isinstance(save, type(None)): 
            plt.savefig(save, bbox_extra_artists=(frame,), bbox_inches = 'tight') 
        plt.show() 


    def Tap_vIPI(self, target, tap = 1, window = None):
        """Plot of tap length and IPI for a specified target.
        
        Parameters
        ----------
        tap : 1 or 2, optional
            Which tap length you want to compare with (first or second tap).
            Defaults to 1.
        target : int
            target IPI
        window : int or None, Optional
            Size of moving average window if you want to do a moving average, must be odd.
            Defaults to None.
        
        Returns
        -------
        out : plt.plot"""

        # define the plotting style 
        plt.style.use('default')
        fig, ax = plt.subplots()
        # blank array for the length of the trials
        length = []
        
        # now that all posibilities of self.rat are lists
        for rat, name in zip(self.rat, self.names):
        
            # define the interval 
            interval = rat.averaged_col('interval', target = target)
            taps = rat.averaged_col(f'tap_{tap}_len', target = target)
            if not isinstance(window, type(None)):
                interval = self.Moving_Average(interval, window)
                taps = self.Moving_Average(taps, window)
            
            # make a list for the trials 
            trials = range(len(interval))
            length.append(trials[-1])

            # plot 
            ax.plot(trials, taps, label= f'{name}, Tap {tap}')
            ax.plot(trials, interval, label=f'{name}, IPI')
        
        # plot a line at where the target should be. 
        ax.hlines(target, 0, np.max(length), colors = ['xkcd:grey'], linestyles = ":", label="Target")

        plt.ylim((0,target +100))
        
        plt.xlabel('Trials', labelpad = -10, loc = "right")
        # code from stackoverflow for formatting the axis as #'s of k 
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self._xlabel))
        
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.1), fancybox=True, ncol=3).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')

        plt.ylabel(' Time (ms)')
        plt.title(f'Tap {tap} & Interval')
        plt.show()


    def IPI(self, target, window = None, ylo = 0, yhi = "Default", save = None):
        """Plots the IPIs for specific target IPI
        
        Parameters
        ----------
        target : int
            Value of target IPI

        window : int or None, optional
            Size of window for moving average if you want one, must be an odd number.
            Defaults to None.

        ylow : int, optional
            Lower limit on y-axis.
            Defaults to 0.
        
        yhi : int, optional
            Upper limit on y-axis
            Defaults to `target` + 100
        
        save : str, optional
            A path to save the figure in, defaults to none.
        
        Returns
        -------
        out : plt.plot"""
    
        # So plots show up on a dark background VSCode
        plt.style.use('default')
        fig, ax = plt.subplots()
        length = []

        # for each of the rats being plotted, 
        for r in range(len(self.rat)): 
            # find the coefficient of variation for this rat and then plot it. 
            interval = self.rat[r].averaged_col('interval', target)
            if not isinstance(window, type(None)):
                interval = self.Moving_Average(interval, window)
            # define the x axis based on the length of the successes
            trials = range(len(interval))
            length.append(trials[-1])
            # plot with a different color for each rat in the ratlist. 
            ax.plot(trials, interval, color = self.farben[r], label=f'{self.names[r]}')
        
        ax.hlines(target, 0, np.max(length), 'xkcd:grey', ":", label = "Target") 

        if yhi == 'Default':
            plt.ylim((ylo, target+100))
        else:
            plt.ylim((ylo, yhi)) 
        
        plt.xlabel('Trials', loc = "right")
        # code from stackoverflow for formatting the axis as #'s of k 
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self._xlabel))
        
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=2).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')
        
        plt.ylabel(f' Interval (miliseconds)')
        plt.title(f'IPI for {target}ms target IPI')
        if not isinstance(save, type(None)):
            fig.savefig(save, bbox_extra_artists=(frame,), bbox_inches = 'tight') 
        plt.show() 


    def Success(self, target, error, window, save = None): 
        """Plots the average percentage of trials within a moving window where the trial IPI was 
        +- error % away from the target IPI. First computes this average for each rat then averages over all rats.
        
        Parameters
        ----------
        target : int
            Value of target IPI

        error : int or float
            The percentage within the target ipi that counts as a success.

        window : int
            Size of moving window.
        
        save : str, optional
            A path to save the figure in, defaults to none.
        
        Returns
        -------
        out : plt.plot"""


        # So plots show up on a dark background VSCode
        plt.style.use('default')
        fig, ax = plt.subplots()

        # for each of the rats being plotted, 
        for r, name in zip(range(len(self.rat)), self.names):
            # find the coefficient of variation for this rat and then plot it. 
            success = self.rat[r].TrialSuccess(error = error, avgwindow = window, target = target)
            
            # define the x axis based on the length of the successes
            trials = range(success.shape[0])
            # plot with a different color for each rat in the ratlist. 
            ax.plot(trials, success, label=f'{name} group', color = self.farben[r])

        # Aesthetic Changes ________________________________________________

        plt.ylim((0,100))
        
        plt.xlabel('Trials', loc = "right")
        # code from stackoverflow for formatting the axis as #'s of k 
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self._xlabel))
        
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=2).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')
        
        plt.ylabel(f'Percent of trials within limit (moving {window} trial window) ')
        plt.title(f'Success Rate within +-{error}% of {target}ms target IPI')
        if not isinstance(save, type(None)):
            fig.savefig(save, bbox_extra_artists=(frame,), bbox_inches = 'tight') 
        plt.show()


    def CV(self, target, window, box = None, ylo = 0, save = None): 
        """Plots a moving window coefficient of variation over all IPIs for a particular target.
        
        Parameters
        ----------
        target : int
            Value of target IPI

        window : int
            Size of window for moving cv, must be an odd number.

        box : int or None, optional
            Size of boxcar if you want to run a boxcar filter over data.
            Defaults to None.

        ylo : int, optional
            Lower limit on y-axis.
            Defaults to 0.
        
        save : str, optional
            A path to save the figure in, defaults to none.
        
        Returns
        -------
        out : plt.plot"""

        
        plt.style.use('default')
        fig, ax = plt.subplots()

        # make an array for the max height
        height = []
        

        for r in range(len(self.rat)): 
            # find the coefficient of variation for this rat and then plot it. 
            cv = self.Moving_CV(self.rat[r].averaged_col('interval', target = target), window = window)
            if not isinstance(box, type(None)):
                cv = self.Boxcar(cv, box)
            trials = range(len(cv))

            max = np.nanmax(cv)
            height.append(max)

            # plot with a different color for each rat in the ratlist. 
            ax.plot(trials, cv, color = self.farben[r], label=f'{self.names[r]} group')

        ylimit = np.max(height) + 0.02
        plt.ylim((ylo,ylimit))
        
        plt.xlabel('Trials', loc = "right")
        # code from stackoverflow for formatting the axis as #'s of k 
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self._xlabel))
        
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=3).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')

        plt.ylabel('Coefficient of Variation')
        plt.title(f'Coefficient of Variation for {target}ms target IPI')
        if not isinstance(save, type(None)):
            fig.savefig(save, bbox_extra_artists=(frame,), bbox_inches = 'tight') 
        plt.show()



class Rat2000:

    def __init__(self, rats, labels):
        self.rats = rats 
        self.labels = labels

        self.farben = ['xkcd:slate grey', 'xkcd:deep green', 'xkcd:greyish green', 'xkcd:cool grey', 'xkcd:slate grey', 'xkcd:deep green', 'xkcd:greyish green', 'xkcd:cool grey']
        self.pattern = ['-', '-', '-', '-', ":", ":", ":", ":"]


    def Plot(self, ptype = "IPI", target = 700, error = 20, save = False):

        # Graph of IPI vs. trials 
        if ptype == ("IPI" or "interval" or "Interval"):
            self.IPI(target, self.labels, save)

        # Graph of Success vs. trials 
        elif ptype == ("Success" or "success"):
            self.Success(target, error, self.labels, save)

        elif ptype == ("CV" or "cv"):
            self.CV(target, self.labels, save)


    def IPI(self, target, labels, save):
        ipis = []
        tar = f'{target}'
        for rat in self.rats:
            ipis.append(rat.df.loc[tar].to_numpy()[0])
        
        plt.style.use('default') 
        
        for i in range(len(self.rats)): 
            plt.bar(i+1, height = self.rats[i].df.loc[tar].to_numpy()[0], label = labels[i], color = self.farben[i])
        
        for i in range(len(self.rats)):
            ys = (self.rats[i].IPIs)[0]
            xs = np.full(len(ys), i+1)
            plt.scatter(xs, ys, s=6, c='black')

        ys = (self.rats[2].IPIs)[1]
        xs = np.full(len(ys), 3)
        plt.scatter(xs, ys, s=6, c='black', label = "Individual Rats' Performance")
        
        plt.xticks(np.arange(len(self.rats))+1, labels)
        plt.ylim((0, target+100))
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=2).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')
        plt.title(f'Mean IPI for last 2000 trials -- {target}ms Target IPI')
        plt.ylabel("Interpress Interval (miliseconds)")
        plt.xlabel("Rat Groups") 

        if save != False:
            plt.savefig(save, bbox_extra_artists=(frame,), bbox_inches = 'tight') 
            plt.show()
        else:
            plt.show()


    def Success(self, target, error, labels, save):
        tar = f'{target}'
        
        plt.style.use('default') 

        for i in range(len(self.rats)): 
            plt.bar(i+1, height = self.rats[i].df.loc[tar].to_numpy()[2], label = labels[i], color = self.farben[i])
        
        for i in range(len(self.rats)):
            ys = (self.rats[i].SRs)[0]
            xs = np.full(len(ys), i+1)
            plt.scatter(xs, ys, s=6, c='black')
        
        ys = (self.rats[2].SRs)[1]
        xs = np.full(len(ys), 3)
        plt.scatter(xs, ys, s=6, c='black', label = "Individual Rats' Performance")

        plt.xticks(np.arange(len(self.rats))+1, labels)
        plt.ylim((0,100))
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=2).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')
        plt.title(f'Mean Success Rate for last 2000 trials -- {target}ms Target IPI')
        plt.ylabel(f'Percent of Rats within {error}% of target IPI')
        plt.xlabel("Rat Groups")

        if save != False:
            plt.savefig(save, bbox_extra_artists=(frame,), bbox_inches = 'tight') 
            plt.show()
        else:
            plt.show()


    def CV(self, target, labels, save):
        cvs = []
        tar = f'{target}'
        for rat in self.rats:
            cvs.append(rat.df.loc[tar].to_numpy()[1])
        
        plt.style.use('default') 
        
        for i in range(len(self.rats)): 
            plt.bar(i+1, height = self.rats[i].df.loc[tar].to_numpy()[1], label = labels[i], color = self.farben[i])
        
        
        for i in range(len(self.rats)):
            ys = (self.rats[i].CVs)[0]
            xs = np.full(len(ys), i+1)
            plt.scatter(xs, ys, s=6, c='black')

        ys = (self.rats[2].CVs)[1]
        xs = np.full(len(ys), 3)
        plt.scatter(xs, ys, s=6, c='black', label = "Individual Rats' last 2000 trial avg")
        
        plt.xticks(np.arange(len(self.rats))+1, labels)
        plt.ylim((0,0.35))
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=2).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')
        plt.title(f'Mean CV for last 2000 trials -- {target}ms Target IPI')
        plt.ylabel("Coefficient of Variation")
        plt.xlabel("Rat Groups") 

        if save != False:
            plt.savefig(save, bbox_extra_artists=(frame,), bbox_inches = 'tight') 
            plt.show()
        else:
            plt.show()


class ChunkyRats:
    # FOR THE ChunkyMonkey class

    def __init__(self, ratdata, ratname, targets):
        """ Class for plotting the rat data. This works on a single rat's data, or a few rat's data. 
        The rats' data are plotted without averaging. 
        Interacts with the DataHolder objects
        
        Parameters 
        ---- 
        ratdata : object 
            Name of the object returned from the DataHolder(presses, session) class
        ratname : string 
            Name of the rat to be displayed on the plots. 
        """

        # check to see if the rat names are in a list. If not, that means there is only one rat name in the list. 
        if not isinstance(ratname, list): 
            # make the item into a "list" so it can be iterated over the same code. 
            self.names = [ratname]       
        else: 
            self.names = ratname 

        # check to see if the rat data is in a list. If not, that means there is only one rat data in the list 
        if not isinstance(ratdata, list):
            # make the item into a "list" so it can be iterated over the same code 
            self.rat = [ratdata] 
        else:
            # if the item is already a list, then make it a self. object
            self.rat = ratdata  
        
        # check to see if the rat data is in a list. If not, that means there is only one rat data in the list 
        if not isinstance(targets, list):
            # make the item into a "list" so it can be iterated over the same code 
            self.targets = [targets] 
        else:
            # if the item is already a list, then make it a self. object
            self.targets = targets  

        self.colors = ['xkcd:slate grey', 'xkcd:deep green', 'xkcd:greyish green', 'xkcd:cool grey', 'xkcd:slate grey', 'xkcd:deep green', 'xkcd:greyish green', 'xkcd:cool grey']

    def _xlabel(self, value, tick_number=None):
        if abs(value) < 1000:
            num = 0 
        else:
            num = ma.floor(ma.log10(abs(value))/3)
        value = round(value / 1000**num, 2)
        return f'{value:g}'+' KM'[num]


    def _pretty(self, ax, targets, ylim = False):
        # Aesthetic Changes ________________________________________________

        if ylim != False:
            plt.ylim((0, ylim)) 
        else: 
            ylim = np.max(targets)+100
            plt.ylim((0,ylim))
        
        plt.xlabel('Trials', labelpad = -10, loc = "right")
        # code from stackoverflow for formatting the axis as #'s of k 
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self._xlabel))
        
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.1), fancybox=True, ncol=3).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')


    def Plot(self, ptype = 'IPI', window = 1000, minwindow = 100, error = 10, boxcar = 300, ymin=0, ymax=900, numsessions = 10, savepath=False):
        """ Returns a plot of the average 1st tap length and average IPI for each session. 
        
        Parameters 
        ---- 
        ptype : str
            REQUIRED
            Name of the plot that you want to make. 
            allowed strings: "IPI", "Success", "Tap1", "Tap2" 
        
        window : int 
            REQUIRED
            Number for the window for moving average calculatons 
            Default is 1000 

        minwindow : int 
            REQUIRED
            Number for the minimum window for moving average calculatons 
            Default is 100
        
        error : int
            Only required for Success plots
            Number for the error margins for the Success plots. 
            Default is 10% 
        
        Returns 
        ---- 
        unnamed : matplotlib plot 
        """
        
        # Graph of IPI vs. trials 
        if ptype == "IPI":
            self.IPI(savepath)

        # Graph of Success vs. trials 
        elif ptype == "Success":
            self.Success(error, savepath)
        
        elif ptype == ("CV" or "cv"):
            self.CV(window, minwindow, boxcar)
   

    def IPI(self, savepath):
        """ Plots the Interpress interval versus the number of trials"""
        
        k = len(self.rat) 
        # define a blank list for finding the shortest Chunk -> will be used for plotting purposes. 
        lens = []
        for r in self.rat:
            for i in range(len(r.intervals)):
                lens.append(((r.Find_IPIDiff())['Mean']).shape[0])
        cutoff = min(lens) 

        # create the blank plot 
        plt.style.use('default')
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)
        fig = plt.figure(figsize=(15,6))
        # define the xs list 
        xs = np.arange(cutoff) 

        for rat, n in zip(self.rat, range(len(self.rat))): 
            # calculate the chunk
            Chunk = rat.Find_IPIDiff()[:cutoff]
            # plot the means 
            plt.bar(xs*(k+1)+n, Chunk['Mean'].to_numpy(), color = self.colors[n], label = self.names[n])
            for i in range(1, Chunk.shape[1]-1):
                # plot each of the individual rats for each chunk average. 
                plt.scatter(xs*(k+1)+n, Chunk[i].to_numpy(), s=6, c='black')
            
        plt.scatter(xs*(k+1)+n, Chunk[0].to_numpy(), s=6, c='black', label = "Individual Rats' Average")

            
        # Relabel the ticks to be the session number. 
        plt.xticks(ticks=xs*(k+1)+(k-1)/2, labels=xs+1) 

        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=4).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')
        title = plt.title(f'IPI Difference between {self.targets[0]}ms & {self.targets[1]}ms target trials ')

        plt.ylabel("Difference (ms)")
        plt.xlabel("Nth group of 1000 trials") 

        # if the savepath is defined, save the figure.
        if savepath != False:
            plt.savefig(savepath, bbox_extra_artists=(frame, title), bbox_inches = 'tight') 
        plt.show()

    def Success(self, error, savepath):
        """ Plots the Success within +- error % of the IPI versus the number of trials
        
        Error = The error # that you gave to the ChunkyMonkey object -- error is included for plot labeling, not for calculations"""
        rat = self.rat[0]

        # define a blank list for finding the shortest Chunk -> will be used for plotting purposes. 
        lens = []
        # find the shortest means list (will be determined by # of sessions the rat had)
        for i in range(len(rat.success)):
            lens.append(((rat.success[i])['Mean']).shape[0])
        cutoff = min(lens)-1

        # create the blank plot 
        plt.style.use('default')
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)
        fig = plt.figure(figsize=(15,6))
        # define the xs list 
        xs = np.arange(cutoff)   

        # for each of the dataframes inside the ChunkyMonkey object, targets, and names,
        for s, name in zip(range(len(rat.success)), self.names):
            # define the chunk we are looking at (will be X00 trial data dataframe)
            chunk = (rat.success[s])[:cutoff]
            # plot the mean IPI within the chunk
            plt.bar(xs*3+s, height = chunk['Mean'].to_numpy(), label = name, color = self.colors[s])
            # for i rats, (the # of columns in the chunk - 1 (for the mean column)) 
            for i in range(chunk.shape[1]-1):
                plt.scatter((xs*3)+s, chunk[f'Rat_{i}_{self.targets[s]}'].to_numpy(), s=6, c = 'black')
        # plot one set again with a label 
        plt.scatter((xs*3)+s, chunk[f'Rat_{i}_{self.targets[s]}'].to_numpy(), s=6, c = 'black', label = "Individual Rats' Performance")

        # Relabel the ticks to be the session number. 
        plt.xticks(ticks=(xs*3)+1/2, labels=xs+1) 
        
        # do plot formatting and aesthetics
        frame = plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=True, ncol=4).get_frame()
        frame.set_edgecolor("black")
        frame.set_boxstyle('square')

        title = plt.title(f'Success Rate of {self.targets[0]}ms & {self.targets[1]}ms target trials ')

        plt.ylabel(f'Percent of trials within {error}% of Target IPI')
        plt.xlabel("Nth group of 1000 trials") 

        # if the savepath is defined, save the figure.
        if savepath != False:
            plt.savefig(savepath, bbox_extra_artists=(frame, title), bbox_inches = 'tight') 
        plt.show()
        plt.show()
