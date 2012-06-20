'''
Created on Jun 20, 2012

@author: jsalvatier
'''
from distutils.core import setup, Extension

module1 = Extension('advinc', sources=['advinc/advinc.c'],
                        include_dirs=[])

setup(name = 'advinc',
        version='1.0',
        description='indexing',
        ext_modules = [module1])