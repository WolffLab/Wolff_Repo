from datetime import timedelta
import os
from Wolff_Repo.Utils.simple import OrdToDate
import pandas as pd
import numpy as np

class matlab_to_csv:

    def __init__(self,press_in,sess_in):
        self.press_in = press_in
        self.sess_in = sess_in
        self.press_out = pd.DataFrame()
        self.sess_out = pd.DataFrame()

    def copy_press_cols(self,cols):
        """directly copy columns from press in to press out"""

        #allows user to input string or list of strings
        if isinstance(cols,str):
            cols = [cols]
        
        for col in cols:
            if not col in self.press_out:
                self.press_out[col] = self.press_in[col]

    def copy_sess_cols(self,cols):
        if isinstance(cols,str):
            cols = [cols]
        for col in cols:
            if not col in self.sess_out:
                self.sess_out[col] = self.sess_in[col]

    #determines what number each press is within that session and adds it as a column
    def get_n_in_sess(self):    
        n_in_sess = []
        for i in set(self.press_in['n_sess']):
            n_in_sess = n_in_sess + list(range(1,1+len(self.press_in.loc[self.press_in['n_sess'] == i])))
        self.press_out['n_in_sess'] = n_in_sess

    def get_interval(self):
        self.press_out['interval'] = self.press_in['tap2Times_on'] - self.press_in['tap1Times_on']

    def get_tap_1_len(self):
        self.press_out['tap_1_len'] = self.press_in['tap1Times_off'] - self.press_in['tap1Times_on']

    def get_tap_2_len(self):
        self.press_out['tap_2_len'] = self.press_in['tap2Times_off'] - self.press_in['tap2Times_on']

    def get_ratio(self):
        self.needs_press(['tap_1_len','interval'])
        self.press_out['ratio'] = self.press_out['tap_1_len']/self.press_out['interval']

    def get_loss(self):
        self.copy_press_cols('n_sess')
        self.copy_sess_cols('target')
        self.needs_press('interval')
        targets = np.array([self.sess_out.iloc[i-1]['target'] for i in self.press_out['n_sess']])
        self.press_out['loss'] = (self.press_out['interval'] - targets)/targets

    def drop_na(self):
        self.press_out = self.press_out.dropna()

    def drop_0(self):
        self.press_out = self.press_out.loc[~(self.press_out == 0).all(axis = 1)]

    def needs_press(self,cols):
        if isinstance(cols,str):
            cols = [cols]
        for col in cols:
            if not col in self.press_out:
                exec(f"self.get_{col}()")

    def needs_sess(self,cols):
        if isinstance(cols,str):
            cols = [cols]
        for col in cols:
            if not col in self.sess_out:
                exec(f"self.get_{col}()")

    def set_types(self,**kwargs):
        for key,val in kwargs.items():
            if key in self.press_out:
                self.press_out[key] = self.press_out[key].astype(val)
            if key in self.sess_out:
                self.sess_out[key] = self.sess_out[key].astype(val)

    def get_starttimes(self):
        #get times for each session
        starttimes = []
        for i in self.sess_in['startTime']:
            try:
                starttimes.append(OrdToDate(i))
            except ValueError:
                starttimes.append(OrdToDate(i,rnd = 'second'))
        self.sess_out['starttime'] = starttimes

    def get_times(self):
        #get times for each press
        self.needs_sess('starttimes')
        times = []
        for i in range(len(self.press_in)):
            row = self.press_in.loc[i]
            sess_time = self.sess_out.loc[int(row['n_sess']-1)]['starttime']
            times.append(sess_time + timedelta(seconds = row['tap1Times_on']/1000))
        self.press_out['time'] = times

    def get_sess_size(self):
        sess_lengths = []
        #get amount in each session
        for n in range(len(self.sess_in)):
            sess_lengths.append(len(self.press_out.loc[self.press_out['n_sess'] == n+1]))
        self.sess_out['sess_size'] = sess_lengths
    
    def drop_sess(self,ns):
        if isinstance(ns,int):
            ns = [ns]
        for n in ns:
            self.sess_out.drop(n-1,inplace=True)
            self.press_out = self.press_out.loc[~(self.press_out['n_sess']==n)]

    #change a target in sesion data
    def change_target(self,old,new):
        targets = []
        for i in self.sess_out['target']:
            if i==old:
                targets.append(new)
            else:
                targets.append(i)
        self.sess_out['target'] = targets

    #add session column for target of next session
    def get_next_target(self):
        self.copy_sess_cols('target')
        self.sess_out['next_target'] =  self.sess_out['target'].to_list()[1:] + [pd.NA]
        
    # add session column for target of previous session
    def get_prev_target(self):
        self.copy_sess_cols('target')
        self.sess_out['prev_target'] =  [pd.NA] + self.sess_out['target'].to_list()[0:-1]

    def save_press(self,pressloc):
        self.press_out.to_csv(pressloc,index = False)
    
    def save_sess(self,sessloc):
        self.sess_out.index+=1
        self.sess_out.to_csv(sessloc,index_label = 'n_sess')






def conv_mat_to_csv(loc,fname):

    if os.path.exists(loc + 'instructions.txt'):
        file = open(loc + 'instructions.txt','r')
        inst = eval(file.read())
        file.close()
        print('instructions found')
    else:
        inst = dict()

    pressloc = loc + fname.replace(".mat","_pressinfo.csv")
    sessloc = loc + fname.replace(".mat","_sessinfo.csv")

    sess_in = pd.read_csv(sessloc)
    press_in = pd.read_csv(pressloc)

    tempclass = matlab_to_csv(press_in,sess_in)
    tempclass.copy_press_cols(['reward','n_sess'])
    tempclass.needs_press(['n_in_sess','interval','tap_1_len','tap_2_len','ratio','loss','times'])
    tempclass.drop_na()
    tempclass.drop_0()

    tempclass.set_types(reward = int,n_sess=int,n_in_sess=int,interval = float,tap_1_len=float,tap_2_len=float,ratio=float)

    tempclass.copy_sess_cols(['target','upper','lower'])
    if 'change_target' in inst:
        tempclass.change_target(inst['change_target'][0],inst['change_target'][1])
        
    tempclass.needs_sess(['starttimes','sess_size','next_target','prev_target'])

    if 'drop_sess' in inst:
        tempclass.drop_sess(inst['drop_sess'])

    
    tempclass.save_press(pressloc)
    tempclass.save_sess(sessloc)