
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[6]:


from IPython.display import display, HTML, Javascript, set_matplotlib_formats
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import sklearn
import statsmodels.api as sm
from joblib import Parallel, delayed
from numpy import inf, arange, array, linspace, exp, log, power, pi, cos, sin, radians, degrees
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style('whitegrid')
set_matplotlib_formats('png', 'pdf')


# In[7]:


def ExcelSaver(df):
    file=input('Document name is: ')
    writer=pd.ExcelWriter(file, engine = 'xlsxwriter')
    df.to_excel(writer)
    writer.save()


# In[8]:


if __name__=='__main__':
    try:    
        get_ipython().system('jupyter nbconvert --to python ImportAll.ipynb')
        # python即转化为.py，script即转化为.html
        # file_name.ipynb即当前module的文件名
    except:
        pass

