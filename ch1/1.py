#coding:utf-8


from random import *
from numpy import *

arr = random.rand(4,4)

print arr

randMat = mat(random.rand(4,4))
print randMat.I #逆矩阵

print randMat*randMat.I #单位矩阵


