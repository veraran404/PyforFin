
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[5]:


def align_figures():
    import matplotlib
    from matplotlib._pylab_helpers import Gcf
    from IPython.display import display_html
    import base64
    from ipykernel.pylab.backend_inline import show

    images = []
    for figure_manager in Gcf.get_all_fig_managers():
        fig = figure_manager.canvas.figure
        png = get_ipython().display_formatter.format(fig)[0]['image/png']
        src = base64.encodestring(png).decode()
        images.append('<img style="margin:0" align="left" src="data:image/png;base64,{}"/>'.format(src))

    html = "{}".format("".join(images))
    show._draw_called = False
    matplotlib.pyplot.close('all')
    display_html(html, raw=True)


# In[6]:


if __name__=='__main__':
    try:    
        get_ipython().system('jupyter nbconvert --to python Functions.ipynb')
        # python即转化为.py，script即转化为.html
        # file_name.ipynb即当前module的文件名
    except:
        pass

