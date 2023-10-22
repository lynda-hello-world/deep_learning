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

class Exp(Function):
    #input:Variable.data
    #output:Variable.data
    def forward(self, x):
        y=np.exp(x)
        return y

#input:Function,Variable,eps=1e-4
#output:number
def numerical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    dy=(y1.data-y0.data)/(2*eps)
    return dy

def f(x):
    A=Square()
    B=Exp()
    C=Square()
    return C(B(A(x)))

x=Variable(np.array(0.5))
dy=numerical_diff(f,x)
print(dy)