import numpy
from cvxpy import *

'''
	minimize -x1^4 -2x2^4 - x3^4 - x1^2x2^2 - x1^2x3^2
	subject to x1^4 + x2^4 + x3^4 - 25 = 0
			   8x1^2 + 14x2^2 + 7x3^2 - 56 = 0

	Answer:
		x* = [1.874605, 0.465820, 1.884720]
		l* = [-1.223464, -0.274937]
		f(x*) = -38.284828

'''

k = 0;
xk = numpy.asmatrix([3.0, 1.5, 3.0]).T;
lk = numpy.asmatrix([-1.0, -1.0]).T;
tolerance = 10e-8;
dx = numpy.asmatrix([1.0, 1.0, 1.0]).T;
while (numpy.linalg.norm(dx, 2) >= tolerance):


	#===Evaluate Lagrangian Hessian===#
	x = numpy.ravel(xk);
	l = numpy.ravel(lk);

	#===Make f===#
	f = -x[0]**4.0 - 2.0*x[1]**4.0 - x[2]**4.0 - (x[0]**2.0)*(x[1]**2.0) - (x[0]**2.0)*(x[2]**2.0);

	#===Print For User===#
	print "K: ", k, " Xk: ", xk.T, " Lk: ", lk.T, " f: ", f;

	#===Make a Vector===#
	a = numpy.asmatrix([x[0]**4.0 + x[1]**4.0 + x[2]**4.0 - 25.0, 8.0*x[0]**2.0 + 14.0*x[1]**2.0 + 7.0*x[2]**2.0 - 56.0]).T;

	#===Make f0 Gradient===#
	grad_f = numpy.asmatrix([ -4.0*x[0]**3.0 - 2.0*x[0]*x[1]**2.0 - 2.0*x[0]*x[2]**2.0, 
			   				  -8.0*x[1]**3.0 - 2.0*(x[0]**2.0)*x[1],
			   				  -4.0*x[2]**3.0 - 2.0*(x[0]**2.0)*x[2] ]).T;

	#===Make a1 Gradient===#
	grad_a1 = numpy.asmatrix([4.0*x[0]**3.0, 4.0*x[1]**3.0, 4.0*x[2]**3.0]).T;

	#===Make a2 Gradient===#
	grad_a2 = numpy.asmatrix([16.0*x[0], 28.0*x[1], 14.0*x[2]]).T;

	#===Make a Gradient Matrix===#
	grad_a = numpy.r_[ grad_a1.T, grad_a2.T];
	
	#===Make f0 Hessian===#
	H_f = [ [ -12.0*x[0]**2.0 - 2.0*x[1]**2.0 - 2.0*x[2]**2.0, -4.0*x[0]*x[1], -4.0*x[0]*x[2] ],
			[ -4.0*x[0]*x[1], -24.0*x[1]**2.0 - 2.0*x[0]**2.0, 0.0 ],
			[ -4.0*x[0]*x[2], 0.0, -12.0*x[2]**2.0 - 2.0*x[0]**2.0 ] ];
	H_f = numpy.asmatrix(H_f);

	#===Make a1 Hessian===#
	H_a_1 = [ [12.0*x[0]**2.0, 0.0, 0.0], [0.0, 12.0*x[1]**2.0, 0.0], [0.0, 0.0, 12.0*x[2]**2.0]];
	H_a_1 = numpy.asmatrix(H_a_1);

	#===Make a2 Hessian===#
	H_a_2 = [ [16.0, 0.0, 0.0], [0.0, 28.0, 0.0], [0.0, 0.0, 14.0]];
	H_a_2 = numpy.asmatrix(H_a_2);

	#===Combine===#
	H_L = H_f - l[0]*H_a_1 - l[1]*H_a_2;

	#===Make System===#
	if (1):
		Zeros = numpy.zeros((2,2));
		KKT = numpy.c_[H_L, -grad_a.T];
		KKT = numpy.r_[KKT, numpy.c_[-grad_a, Zeros]];	
		LHS = numpy.r_[-1.0*(grad_f - l[0]*grad_a1 - l[1]*grad_a2), a];	

		#===Solve===#
		sol = numpy.linalg.solve(KKT, LHS);
		dx = sol[0:3]; dl = sol[3::];

		#===Update===#
		xk = xk + dx;
		lk = lk + dl;

	else:

		#===Force HL PSD===#
		lamb, v = numpy.linalg.eig(H_L);
		L = numpy.diag(numpy.abs(lamb));
		H_L = v * L * v.T 

		#===Create Linear Cost===#
		c = Parameter(3);
		c.value = grad_f;

		del_x = Variable(3);
		G_A = Parameter(2, 3)
		G_A.value = grad_a;
		constraints = [G_A*del_x == -a];

		#===Create And Solve Problem===#
		obj = Minimize(0.5*quad_form(del_x, H_L) + c.T*del_x);
		prob = Problem(obj, constraints);
		prob.solve();
		dx = prob.variables()[0].value;

		#===Find Eigenvalue===#
		lhs = H_L*dx + grad_f
		sol = numpy.linalg.lstsq(grad_a.T, lhs);
		
		#===Update===#
		xk = xk + dx;
		lk = sol[0];
		
	#===Update===#
	k = k +1;

