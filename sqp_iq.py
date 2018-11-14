import numpy
from cvxpy import *

'''
	minimize || x12 - x34 ||_{2}^{2}
	subject to -x12^T H12 x12 + c12^T x12 + g12 >=0
			   -1/8 * x34^T H34 x34 + c34^T x34 + g34 >=0
	Where:
			x12 = [x1] 			x34 = [x3]
				  [x2]		 		  [x4]

			H12 = [1/4 0]		H34 = [5 3]
				  [0   1]			  [3 5]

			c12 = [1/2]			c34 = [11/2]
				  [ 0 ]				  [13/2]
			
			g12 = 3/4			g34 = -35/2

	Answer:

			x* = [x12, x34]^T = [2.044750, 0.852716, 2.544913, 2.485633]^T
			u* = [0.957480, 1.100145]^T
			f(x*) = 2.916560
'''

k = 0;
xk = numpy.asmatrix([1.0, 0.5, 2.0, 3.0]).T;
uk = numpy.asmatrix([1.0, 1.0]).T;
tolerance = 10e-5;
dx = numpy.asmatrix([1.0, 1.0, 1.0, 1.0]).T;
while (numpy.linalg.norm(dx, 2) >= tolerance):

	#===Split===#
	x12 = xk[0:2];
	x34 = xk[2:4];

	#===Make Accessible===#
	x = numpy.ravel(xk);
	u = numpy.ravel(uk);

	#===Make f===#
	f = (1.0*(x[0]-x[2])**2.0 + 1.0*(x[1]-x[3])**2.0);

	#===Make grad f===#
	grad_f = numpy.asmatrix([x[0]-x[2], x[1]-x[3], x[2]-x[0], x[3]-x[1]]).T;

	#===Make Hessian f===#
	H_f = numpy.asmatrix([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]]);

	#===Print For User===#
	print "K: ", k, " Xk: ", xk.T, " Uk: ", uk.T, " f: ", f;

	#===Make c1 Constraint===#
	H12 = numpy.asmatrix([[1.0/4.0, 0.0],[0.0,1.0]]);
	c12 = numpy.asmatrix([1.0/2.0, 0.0]).T; g12 = 3.0/4.0;
	c1 = -1.0*x12.T*H12*x12 + c12.T*x12 + g12;

	#===Make c2 Constraint===#
	H34 = numpy.asmatrix([[5.0, 3.0],[3.0,5.0]]);
	c34 = numpy.asmatrix([11.0/2.0, 13.0/2.0]).T; g34 = -35.0/2.0;
	c2 = (-1.0/8.0)*x34.T*H34*x34 + c34.T*x34 + g34;	

	#===Make c Vector===#
	c = numpy.asmatrix([numpy.ravel(c1)[0], numpy.ravel(c2)[0]]).T;
	
	#===Make grad_c===#
	grad_c1 = numpy.asmatrix(numpy.r_[numpy.ravel([ -2.0*H12*x12 + c12]), 0.0, 0.0]).T;
	grad_c2 = numpy.asmatrix(numpy.r_[0.0, 0.0, numpy.ravel([ -(1.0/4.0)*H34*x34 + c34])]).T;
	grad_c = numpy.r_[grad_c1.T, grad_c2.T];

	#===Make Hessian c===#
	Zeros = numpy.asmatrix([[0.0,0.0],[0.0,0.0]]);
	H_c1 = numpy.asmatrix(numpy.r_[numpy.c_[-2.0*H12, Zeros], numpy.c_[Zeros, Zeros]]);
	H_c2 = numpy.asmatrix(numpy.r_[numpy.c_[Zeros, Zeros], numpy.c_[Zeros, (-1.0/4.0)*H34]]);

	#===Make Yk===#
	Yk = H_f - u[0]*H_c1 - u[1]*H_c2;

	#===Setup Problem===#
	cost = Parameter(4);
	cost.value = grad_f;
	del_x = Variable(4);
	G_C = Parameter(2, 4)
	G_C.value = grad_c;
	constraints = [G_C*del_x >= -c];

	#===Create And Solve Problem===#
	obj = Minimize(0.5*quad_form(del_x, Yk) + cost.T*del_x);
	prob = Problem(obj, constraints);
	prob.solve();
	dx = prob.variables()[0].value;

	#===Find Eigenvalue===#
	lhs = Yk*dx + grad_f
	sol = numpy.linalg.lstsq(grad_c.T, lhs);

	#===Update===#
	xk = xk + dx;
	uk = sol[0];
	k = k + 1;

