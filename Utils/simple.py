import pandas as pd
import numpy as np
from pandas import NA as nan
import tkinter as tk
import datetime
from tkinter import filedialog
from scipy import stats

def isodd(n):
    """Determine if an integer is odd.
    
    Parameters
    ----------
    n : int
        Value to check.
        
    Returns
    -------
    out : bool
        Wether or not `n` is odd.
        
    Raises
    ------
    TypeError 
        Raised if `n` is not int."""
    
    if not isinstance(n,int):
        raise TypeError(f'type {type(n)} is invalid, must be integer')
    return n%2 > 0

def rndnt(n):
    """Round a value, 0.5 rounds up to 1.
        
    Parameters
    ----------
    n : numeric
        value to round.

    Returns
    -------
    out : int
        Rounded value."""

    return int(np.floor(n+0.5))

def ceil(n):
    """Return closest integer rounding up.
        
    Parameters
    ----------
    n : numeric
        value to round.

    Returns
    -------
    out : int
        Rounded value."""
    return int(np.ceil(n))

def fnt(n):
    """Return closest integer rounding down.
        
    Parameters
    ----------
    n : numeric
        value to round.

    Returns
    -------
    out : int
        Rounded value."""
    return int(np.floor(n))

def totuple(a):
    """Changes an array into a nested tuple.
    
    Parameters
    ----------
    a : np.ndarray
        array to be changed
        
    Returns
    -------
    out : tuple
    """
    if isinstance(a,np.ndarray):
        return tuple(totuple(i) for i in a)
    else:
        return a

def get_file(prompt = 'chose a file'):
    """Promts user to select a file via a gui.
    
    Parameters
    ----------
    prompt : str, optional
        This will be the name of the gui window"""
    #establish a root ui and remove the window
    root = tk.Tk()
    root.withdraw()

    #open file explorer
    loc = filedialog.askopenfilename(title = prompt)

    #return file location as string
    return loc

def get_directory(prompt = 'chose a directory'):
    """Prompts user to select a directory via a gui.
    
    Parameters
    ----------
    prompt : str, optional
        This will be the name of the gui window"""
    #establish a root ui and remove the window
    root = tk.Tk()
    root.withdraw()

    #open file explorer
    loc = filedialog.askdirectory(title = prompt)

    #return file location as string
    return loc
def func_on_subsets(vals, divsize, func, manage_extras = "auto", xs = None):
    """apply a function to subdivisions of size `divsize` of a larger list.
    
    Parameters
    ----------
    vals : iterable
        list of values to apply function on subdivisions of.

    func : function
        function to convolve over `vals`
    
    divsize : int
        Size of each subset

    manage_extras : str, optional
        How to approach if list does not subdivide exactly into subdivisions.
            "auto" - keep last subdivision if size is <divsize/2
            "keep" - keep last subdivision no matter the size
            "drop" - drop last subdivision no matter the size

    xs : list, optional
        x values of vals, if included then `outxs` will be included in return value.

    Returns
    -------
    outlist : list
        List of `func` applied to each subdivision of `vals`.

    outxs : list, optional
        List of x values corresponding to central x value of each subdivision."""

    #if list does not subdivide exactly chose what to do with extra value
    manage_extras = {"auto":rndnt,"keep":ceil,"drop":fnt}[manage_extras]
    #compute the unsupplied value or give error if both or neither were given
    n = manage_extras(len(vals)/divsize)

    outlist = []
    if not isinstance(xs,type(None)):
        outxs = []
    for i in range(n):
        #starting index
        start = i*divsize
        #ending index
        end = min((i+1)*divsize,len(vals))
        #add computed value to list
        outlist.append(func(vals[start:end]))
        if not isinstance(xs,type(None)):
            outxs.append(xs[rndnt((start+end)/2) - 1])
        
    if not isinstance(xs,type(None)):
        return outlist,outxs

    return outlist

#convolve a function over a list you can include a list of x values to get those returned
def func_convolve(vals,func,divsize,xs = None):
    """Convolve a function over a list.
    
    Parameters
    ----------
    vals : iterable
        list of values to convolve function over.

    func : function
        function to convolve over `vals`
    
    divsize : int
        Size of convolution window

    xs : list, optional
        x values of vals, if included then `outxs` will be included in return value.

    Returns
    -------
    outlist : list
        Result of convolution.

    outxs : list, optional
        List of x values corresponding to central x value of each value from the convolution.
        
    Raises
    ------
    ValueError 
        Raised if `divsize` is not odd."""
    
    if not isodd(divsize):
        raise ValueError('divsize must be odd')
    outlist = [func(vals[i:i+divsize]) for i in range(0,len(vals)-divsize+1)]
    if not isinstance(xs,type(None)):
        start =int((divsize/2) - 0.5)
        end = int((-divsize/2) + 0.5)
        outxs = xs[start:end]
        return outlist, outxs
    return outlist

#returns the variation of each  of n sub-divisions of a dataset, variation values are normalized using the variation of the whole set
def sub_variation(vals,n):
    set_var = stats.variation(vals)
    divsize = rndnt(len(vals)/n)
    return func_on_subsets(vals,divsize,lambda x:stats.variation(x)/set_var)

#simple boxcar filter
def boxcar(vals,divsize,**kwargs):
    """Applies a boxcar mean to a list of values.
        
    Parameters
    ----------
    vals : iterable
        values to apply boxcar on.

    divsize : int
        Size of each boxcar.

    **kwargs : see func_on_subsets

    Returns
    -------
    out : int
        Result of applied boxcar."""
    return func_on_subsets(vals,divsize,np.mean,**kwargs)

#performs a linear fit and returns the datapoints and the slope/intercept
def lin_fit(ys,xs = "Default"):
    if xs == "Default":
        xs = range(len(ys))
    m,b=np.polyfit(xs,ys,1)
    fitys = m*xs + b
    return xs,fitys,m,b

#return the slope of convergence subdivisions in a dataset
def var_convergence(vals,n = 4):
    sub_vars = sub_variation(vals,n)
    m = lin_fit(sub_vars)[2]
    return m

#percentage of values within a range
def perc_within(vals,lower,upper):
    """Get percentage of values within a range.
    
    Parameters
    ----------
    vals : iterable
        List of values to pull from.

    lower : numeric
        lower bound.

    upper : numeric
        upper bound
        
    Returns
    -------
    out : float
        percentage of values in `vals` that fall between `upper` and `lower`."""
    
    n = len(vals)
    s = sum(1 for i in vals if upper>=i >= lower)
    return s/n

def sreplace(instr, repl, replwith, max_iterations=100, count_as_alpha = []):
    """replaces all instances of `repl` inside `instr` with `replwith`, checks that repl is not part of a larger word.
    
    Parameters
    ----------
    instr : str
        string to perform replacements in.
    repl : str
        string to replace.
    replwith : str
        what to replace with
    max_iterations : int, optional
        function will loop through string and make replacements this many times. 
        This argument is mainly to avoid problems if `replwith` is inside `repl` as this will cause an infinite loop.
    count_as_alpha : list, optional
        Characters that count as letters, normally only a-z counts.
    
    Returns
    -------
    outstr : str
        A version of `instr` where all versions of `repl` that are not part of larger words get replaced with `replwith`."""
    
    checkalpha = lambda x:(x.isalpha() or (x in count_as_alpha))
    #add that blank incase the thing i want to replace is at the end
    outstr = instr + ' '
    strt = 0
    i=0
    #keep looping if thing needing to be replaced is still in the string
    while outstr[strt:].find(repl)!=-1:
        i+=1

        #the first index of repl in the part of the string that hasnt yet been looked at
        strt = outstr[strt:].index(repl) + strt
        #the last index of the thing
        end = strt + len(repl)

        #check if word is part of another word
        if not (checkalpha(outstr[strt-1]) or checkalpha(outstr[end])):

            #make the replacement
            outstr = outstr[0:strt] + replwith + outstr[end:]
            #move start location for next pass
            strt = strt + len(replwith)
        else:
            strt = end

        if i>=max_iterations:
            raise Exception('timed out')

    # the [0:-1] removes that space i put at the end
    return outstr[0:-1]

def str_in_list(arr, substr):
    """ Returns index of first string in list that contains substr.
    
        Parameters
        ----------
        arr : list
            list to find index within.
        substr : string or list
            string or list of strings to find inside part of list .

        Returns
        -------
        out : int
            first index where all substr are found.
        """
    #make iterable if only one string is input
    if isinstance(substr,str):
        substr = [substr]
    
    #iterate through each string inside array, return index if all substrings are found
    i = 0
    for val in arr:
        if all(item in val for item in substr):
            return i
        i+=1

def take_inner(instr, strtstr = None, endstr = None):
    """ Return string between strt and end in larger string

    Parameters
    ----------
    instr : str
        bigger string to pull from.
    strt : str, optional
        String to take all characters after.
    end : str, optional
        String to take all characters before.

    Returns
    -------
    out : str
        all characters between start and end.
    """
    if isinstance(strtstr,type(None)):
        strt = 0
    elif isinstance(strtstr,str):
        strt = instr.index(strtstr) + len(strtstr)
    
    if isinstance(endstr,type(None)):
        end = len(instr) + 1
    elif isinstance(endstr,str):
        end = instr[strt:].index(endstr) + strt

    return instr[strt:end]

def add_blank_row(df):
    """Add blank row (row of nan values) to a dataframe

    Parameters
    ----------
    df : pd.Dataframe
        dataframe to add a blank row to.

    Returns
    -------
    out : pd.Dataframe
        same dataframe with a blank row added.
    """
    return df.append(pd.Series([nan for _ in range(len(df.columns))],index = df.columns),ignore_index=True)

def row_is(row, params):
    """Check if a row from a dataframe has specific values in its' columns.

    Parameters
    ----------
    row : pd.Series
        row to check to values of.
    params : dict
        A dictionary where the keys are the columns and the values are what the values should be for those keys. You don't need to specify a value for every column in the row.

    Returns
    -------
    out : bool
        True if all values match, otherwise false.
    """
    #iterate through each column specified in the dictionary
    for key,val in params.items():
        #if and value does not match then return false
        if not row[key] == val:
            return False
    #if no falses are returned that all rows match and we can return true
    return True

def digitnum(n,digits = 3):
    """Returns integer as string with 0s in front up till a certain number of digits, for example digitnum(3,3) will return '003' while digitnum(30,3) will return '030'.

    Parameters
    ----------
    n : int
        original integer.
    digits : int,optional
        amount of digits of final string.

    Returns
    -------
    out : str
        number with 0s in front
    """
    out = str(n)
    while len(out) < digits:
        out = '0' + out
    return out

def ynprompt(prompt):
    """Ask user a yes or no question.
    
    Parameters
    ----------
    prompt : question to ask.
    
    Returns
    -------
    out : bool
        True if user answers yes or false if user answers no."""
    while True:
        ans = input(prompt).lower()
        if ans == 'yes' or ans == 'y':
            return True
        elif ans == 'no' or ans == 'n':
            return False

def datasimplify(df):
    """Changes the datatypes in a dataframe to the simplest possible datatype.
    
    Parameters
    ----------
    df : pd.Dataframe
    
    Returns
    -------
    outdf : pd.Dataframe
        new dataframe with columns of simplest datatype"""
    
    outdf = df.copy()
    for col in df.columns:
        try:
            outdf[col] = pd.to_numeric(outdf[col],downcast='unsigned')
        except:
            outdf[col] = outdf[col].astype(str)
    return outdf

#convert from matlab ordinal date to normal date
def OrdToDate(ord,rnd = 'hour'):
    """Convert from matlab ordinal date to normal date.
    
    Parameters
    ----------
    ord : numeric
        Matlab ordinal date to be converted.
    rnd : str, optional
        Acceptable values include 'hour', 'minute', and 'second'. 
        Default is 'hour'.
    Returns
    -------
    out : datetime.Datetime 
    """
    outdate = datetime.datetime.combine(datetime.datetime.min.date(), datetime.datetime.min.time())
    outdate = outdate + datetime.timedelta(days =(ord - 367) )
    return rnddate(outdate,rnd)

#round a date to the nearest value specified by rnd
def rnddate(date,rnd = 'hour'):
    """Round a date to the nearest value specified by `rnd`.
    
    Parameters
    ----------
    date : datetime.datetime
        Date to be rounded
    rnd : str, optional
        Nearest increment to round to. Can be second, hour, or minute. Defaults to hour."""
    outdate = date
    if rnd  in ['second','hour','minute']:
        newseconds = rndnt(outdate.second + (outdate.microsecond * 1e-6))
        outdate = outdate.replace(second = 0,microsecond=0) + datetime.timedelta(seconds = newseconds)
    if rnd  in ['hour','minute']:
        newminutes = rndnt(outdate.minute + (outdate.second/60))
        outdate = outdate.replace(minute = 0,second=0) + datetime.timedelta(minutes=newminutes)
    if rnd  in ['hour']:
        newhours = rndnt(outdate.hour + (outdate.minute/60))
        outdate = outdate.replace(hour = 0,minute=0) + datetime.timedelta(hours = newhours)
    return outdate