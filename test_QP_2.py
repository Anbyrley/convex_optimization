from cvxpy import *
import numpy

'''
	minimize 0.5*(x1^2 + x2^2 + x3^2) + 0.55*x3
	subject to x1 + x2 + x3 = 1
			   x1 >= 0
			   x2 >= 0
			   x3 >= 0

x* = [0.5, 0.5, 0.0]  f* = 0.25


In standard form:
	minimize 0.5*x.THx + c.Tx
	subject to a.T x = b
			   x >= 0

Where:
	H = [1.0, 0.0,   0.0   ]
		[0.0, 1.0,   0.0   ]
		[0.0, 0.0, 1.0/10.0]
	c.T = [0.0, 0.0, 0.55]
	a.T = [1.0, 1.0, 1.0]
	b = 1.0

'''

#===Set Precision===#
numpy.set_printoptions(precision = 3);

#===Create Variable===#
x = Variable(3);

#===Create Hessian===#
H = numpy.asmatrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0/10.0]]);
D = numpy.asmatrix([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0/20.0]]);
D_12 = numpy.linalg.cholesky(D);

#===Create Linear Cost===#
c = Parameter(3);
c.value = numpy.matrix([0.0, 0.0, 0.55]).T;

#===Create Equality Constraint System===#
a = Parameter(3);
a.value = numpy.array([1.0, 1.0, 1.0]);
b = Parameter(1);
b.value = 1.0;

#===Create Constraint System===#
constraints = [(a.T*x == b), (x >= 0)];

#===Create And Solve Problem===#
obj = Minimize(0.5*quad_form(x, H) + c.T*x);
#obj = Minimize(quad_form(x, D) + c.T*x);
prob = Problem(obj, constraints);
prob.solve();

print "f* =", obj.value
xopt = prob.variables()[0].value;
print "x* =",xopt


