from cvxpy import *
import numpy

'''

	minimize 1/2x^T*P*x + q^T*x
	subject to Ax=b

where:
	P = [2 1]	q = [-6]	A = [1 0]	b = 3
		[1 2]		[ 7]

Solution:
	[P A^T] [x] = [-q]
	[A 0  ] [v]	  [ b]

	x* = [3,-5]  v* = 5

'''

#===Form the KKT Matrix===#
H = numpy.matrix([[2.0,1.0],[1.0,2.0]]);
A = numpy.matrix([1.0,0.0]);
KKT = numpy.hstack((H,A.T));
A1 = numpy.matrix([1.0,0.0,0.0]);
KKT = numpy.vstack((KKT, A1));

#===Make Right Hand Side===#
b = [3.0];
q = numpy.matrix([-6.0, 7.0]).T;
k = numpy.vstack((-q, b));

#===Solve===#
opt = numpy.linalg.solve(KKT, k);
x = opt[0:2];
v = opt[2];

#===Print===#
print "X*: ",x.T;
print "V*: ",v;

