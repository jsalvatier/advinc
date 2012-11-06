from numpy import *
from advinc import *



a = arange(12).reshape((3,4)).astype(float)
index = ([1,1,2,0], [0,0,2,3])
vals = [50,50, 30,16]

inplace_increment(a, index, vals)

print a



b = arange(6).astype(float)
index = (array([1,2,0]),)
vals = [50,4,100.1]

inplace_increment(b, index, vals)

print b


