#coding:utf-8

from mrjob.protocol import JSONProtocol
from numpy import *

fw=open('kickStart2.txt', 'w')
for i in [1]:
    for j in range(100):
        fw.write('["x", %d]\n' % random.randint(200))
fw.close()