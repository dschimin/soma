""" Convenience functions for retrieving toy datasets """

import sys

try:
    # Python 2
    from urllib2 import urlopen, URLError
except ImportError:
    # Python 3
    from urllib.request import urlopen
    from urllib.error import URLError

try:
    # Python 2
    from cStringIO import StringIO
except:
    # Python 3
    from io import StringIO

from IPython.display import display
import numpy as np

class APWException(object):

    def __init__(self, text):
        self.text = text

    def _repr_html_(self):
        return "<span style='color: red; font-weight: bold;'>{}</span>".format(self.text)

def get_data(url):
    try:
        data = np.loadtxt(urlopen(url))
    except URLError:
        ex = APWException("Unable to reach server! Are you sure you're connected "
                          "to the internet?")
        display(ex)
        return None

    return data.T


def get_data1():
    return get_data("http://www.adrian.pw/scr/mcmc_data1.txt")

def get_data2():
    return get_data("http://www.adrian.pw/scr/mcmc_data2.txt")
