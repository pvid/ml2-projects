from math import exp, log, tanh, cosh
from functools import partial


class Variable:

    def __init__(self, value, children=[], prop_func=lambda x: None):
        self.value = value
        self.children = children
        self.d = Derivative(prop_func)

    def __str__(self):
        if self.children == []:
            return("Variable with value {}".format(self.value))
        else:
            return("Expression with value {}".format(self.value))

    def backward(self):
        self.d.value = 1
        self.propagate()

    def propagate(self):
        self.d.propagation_function(self)
        for child in self.children:
            child.propagate()

    def __add__(self, b):
        other = convert_to_variable(b)
        return Variable(self.value + other.value,
                        children=[self, other],
                        prop_func=add_prop)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        other = convert_to_variable(b)
        return Variable(self.value * other.value,
                        children=[self, other],
                        prop_func=mul_prop)

    def __rmul__(self, b):
        return self.__mul__(b)

    def __sub__(self, b):
        other = convert_to_variable(b)
        return Variable(self.value - other.value,
                        children=[self, other],
                        prop_func=sub_prop)

    def __rsub__(self, b):
        other = convert_to_variable(b)
        return Variable(other.value - self.value,
                        children=[other, self],
                        prop_func=sub_prop)

    def __truediv__(self, b):
        other = convert_to_variable(b)
        return Variable(self.value / other.value,
                        children=[self, other],
                        prop_func=div_prop)

    def __rtruediv__(self, b):
        other = convert_to_variable(b)
        return Variable(other.value / self.value,
                        children=[other, self],
                        prop_func=div_prop)

    def __getattr__(self, name):
        if name in functions_derivatives.keys():
            function, derivative = functions_derivatives[name]
            prop_function = partial(function_prop_template, derivative)
            return lambda: Variable(function(self.value),
                                    children=[self],
                                    prop_func=prop_function)
        else:
            msg = "'Variable' object has no attribute '{}'".format(name)
            raise AttributeError(msg)


class Derivative:
    def __init__(self, prop_func=None):
        self.value = 0
        self.propagation_function = prop_func


def convert_to_variable(i):
    if type(i) != Variable:
        return Variable(i)
    else:
        return i


def add_prop(var):
    # Sum rule
    var.children[0].d.value += var.d.value
    var.children[1].d.value += var.d.value


def mul_prop(var):
    # Product rule
    var.children[0].d.value += var.d.value * var.children[1].value
    var.children[1].d.value += var.d.value * var.children[0].value


def sub_prop(var):
    # Sum rule in pink
    var.children[0].d.value += var.d.value
    var.children[1].d.value -= var.d.value


def div_prop(var):
    # Division rule
    var.children[0].d.value += var.d.value/var.children[1].value
    var.children[1].d.value -= (var.d.value*var.children[0].value)/(var.children[1].value**2)


def function_prop_template(derivative, var):
    # Chain rule
    var.children[0].d.value += var.d.value * derivative(var.children[0].value)


functions_derivatives = {
    'exp': (exp, exp),
    'log': (log, lambda x: 1/x),
    'tanh': (tanh, lambda x: 1/(cosh(x)**2))
}
