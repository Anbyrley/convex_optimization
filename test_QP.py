from cvxpy import *
import numpy

'''
	minimize x1^2 + 2x2^2 + 3x3^2 + x1x3 + x1 - 2x2 + 4x3
	subject to  3x1 + 4x2 - 2x3 <= 10
				-3x1 + 2x2 + x3 >= 2
				2x1 + 3x2 + 4x3 = 5
				0 <= x1 <= 5
				1 <= x2 <= 5
				0 <= x3 <= 5

	x* = [0.290, 1.413, 0.045]  f* = 1.741

In standard form:

	minimize 1/2 x^T H x + c^T x 
	subject to Gx <= h
			   Ax = b
			   x >= t1
			   -x >= t2
	With:
		H = [ 2 0 1 ]  c^T = [1,-2,4]
			[ 0 2 0 ]	   
			[ 1 0 6 ]
		G = [ 3  4 -2 ]		h^T = [10,2]
			[ 3 -2 -1 ] 	
		a^T = [1,3,4]	b = [5]
		t1^T = [0,1,0]	t2 = [-5,-5,-5]

'''

#===Set Precision===#
numpy.set_printoptions(precision = 3);

#===Create Variable===#
x = Variable(3);

#===Create Hessian===#
H = numpy.asmatrix([[2.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 6.0]]);
D = numpy.asmatrix([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]]);
D_12 = numpy.linalg.cholesky(D);

#===Create Linear Cost===#
c = Parameter(3);
c.value = numpy.matrix([1.0, -2.0, 4.0]).T;

#===Create Inequality Constraint System===#
G = Parameter(2, 3)
G.value = numpy.asmatrix([[3.0, 4.0, -2.0], [3.0, -2.0, -1.0]]);
h = Parameter(2);
h.value = numpy.matrix([10.0, -2.0]).T;

#===Create Equality Constraint System===#
a = Parameter(3);
a.value = numpy.array([2.0, 3.0, 4.0]);
b = Parameter(1);
b.value = 5.0;

#===Create Box Constraints===#
t1 = Parameter(3);
t1.value = numpy.array([0.0, 1.0, 0.0]);
t2 = Parameter(3);
t2.value = numpy.array([-5.0, -5.0, -5.0]);

#===Create Constraint System===#
constraints = [(G*x < h), (a.T*x == b), (x >= t1), (-x >= t2)];

#===Create And Solve Problem===#
#obj = Minimize(0.5*quad_form(x, H) + c.T*x);
#obj = Minimize(quad_form(x, D) + c.T*x);
obj = Minimize((pnorm(D_12*x,2.0)**2.0) + c.T*x);
prob = Problem(obj, constraints);
prob.solve();

print "f* =", obj.value
xopt = prob.variables()[0].value;
print "x* =",xopt
	
