'''
Created on Jun 20, 2012

@author: jsalvatier
'''
from distutils.core import setup, Extension
from numpy import get_include

module1 = Extension('advinc', sources=['advinc/advinc.c'],
                        include_dirs=[get_include()])

setup(name = 'advinc',
        version='1.0',
        description='indexing',
        ext_modules = [module1])