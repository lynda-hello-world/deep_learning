import numpy as np
class Variable:
    def __init__(self,data):
        self.data=data #ndarry
        self.grad=None #ndarry
        self.creator=None #Function
    def set_creator(self,func):
        self.creator=func
    def backward(self):
        #recursion
        '''f=self.creator
        if f is not None:
            x=f.input
            x.grad=f.backward(self.grad)
            x.backward() #recursion'''
        #loop
        funcs=[self.creator]
        while funcs:
            f=funcs.pop()
            x,y=f.input,f.output
            x.grad=f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    #object:get data from Variable
    #       put calculation result into Variable
    #input:Variable
    #output:Variable
    #instance variables:input
    def __call__(self,input):
        x=input.data #get data from Variable
        y=self.forward(x)
        output=Variable(y) # put calculation result into Variable
        output.set_creator(self)
        self.input=input
        self.output=output
        return output
    #object:calculation
    #input:Variable.data
    #output:Variable.data
    def forward(self,x):
        raise NotImplementedError
    
    #object:
    #input:Variable.grad
    #output:Variable.grad
    def backward(self,gy):
        raise NotImplementedError
    
class Square(Function):
    #object:calculation
    #input:Variable.data
    #output:Variable.data
    def forward(self,x):
        y=x**2
        return y
    #object:
    #input:Variable.grad
    #output:Variable.grad
    def backward(self, gy):
        x=self.input.data
        gx=2*x*gy
        return gx

class Exp(Function):
    #input:Variable.data
    #output:Variable.data
    def forward(self, x):
        y=np.exp(x)
        return y
    #input:Variable.grad
    #output:Variable.grad
    def backward(self, gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx

#input:Function,Variable,eps=1e-4
#output:number
def numerical_diff(f,x,eps=1e-4):
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    dy=(y1.data-y0.data)/(2*eps)
    return dy


### check ###
A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)

y.grad=np.array(1.0)
y.backward()
print(x.grad)