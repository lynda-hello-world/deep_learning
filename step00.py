import numpy as np
class Variable:
    def __init__(self,data):
        self.data=data

class Function:
    #object:get data from Variable
    #       put calculation result into Variable
    #input:Variable
    #output:Variable
    def __call__(self,input):
        x=input.data #get data from Variable
        y=self.forward(x)
        output=Variable(y) # put calculation result into Variable
        return output
    #object:calculation
    #input:Variable.data
    #output:Variable.data
    def forward(self,x):
        raise NotImplementedError
    
class Square(Function):
    #object:calculation
    #input:Variable.data
    #output:Variable.data
    def forward(self,x):
        y=x**2
        return y

x=Variable(np.array(10))
f=Square()
y=f(x)
print(y.data)