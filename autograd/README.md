# Autograd project

A very simple autograd engine based on specification found
[here](https://usamec.github.io/ml2/hw1) on the course page.

Grading depends on passing unittests. The test files included with the assignment can be found in the 'tests' subfolder.

To run the tests:

    $ python -m unittest discover .

### Usage

The whole functionality is built into the Variable class.

By initializing Variables and combining them by applying arithmetic operations and functions a DAG is built representing a mathematical expression.

By calling the .backward() method on any of the expressions X (represented by Variable objects) the derivatives of all the
other expressions w.r.t. to the expression X.

It is assumed that the value of a Variable does not change after its creation and that the .backward() method is called only on one of the nodes of a the DAG.

### Comments

Frankly speaking, the implementation is not very good.

The main issue is lack of robustness:
* Changing a value of a Variable does not propagate to expressions (Variables) built on top of it.
* After call the .backward() method on an expression, you cannot call it on any subexpression and get right answers.

Both of these issues could be resolved by adding methods that would facilitate the communication between the nodes of the expression graph. However, that would overly complicate what was to be very simple project.

**Quick note:** What I call 'children' in my implementation
should be more appropriately be called 'parents'
