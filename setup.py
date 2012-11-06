from distutils.core import setup
from distutils.extension import Extension
import numpy 

ext_modules = [
               Extension("advinc", ["advinc/advinc.c"],  include_dirs=[numpy.get_include()]   )]

setup(
  name = 'advinc',
  version = '0.2',
  ext_modules = ext_modules
)
