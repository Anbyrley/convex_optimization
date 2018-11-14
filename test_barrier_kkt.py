from cvxpy import *
import numpy


'''
	minimize t(0.5*x1**2 + 0.5*x2**2 + 0.05*x3**2 + 0.55x3) - log(x1) - log(x2) - log(x3)
	subject to [1.0 1.0 1.0][x1] = 1.0
					  		[x2]
					  		[x3]

'''

#===Set EPS Fix===#
eps = 10e-16;

def find_alpha(xk, dk, t, num_times):
	#===Line Search For Alpha===#
	alphas = numpy.linspace(0, 1, num_times);
	best_value = 100000000.0; current_alpha = 1.0;	
	for this_alpha in alphas:
		xk_new = xk + this_alpha*dk;
		xk_new = numpy.asarray(xk_new);
		xk_new += eps;		
		current_value = t*(0.5*(xk_new[0]**2.0 + xk_new[1]**2.0 + 0.1*xk_new[2]**2.0) + 0.55*xk_new[2]); 
		current_value += (-numpy.log(xk_new[0]) -numpy.log(xk_new[1]) -numpy.log(xk_new[2]));
		if current_value < best_value:
			best_value = current_value;
			current_alpha = this_alpha;
	return current_alpha;


def create_KKT_system( H, A, grad_f ):

	#===Make KKT Shape===#
	KKT_rows = H.shape[0] + A.shape[0];
	KKT_cols = H.shape[1] + A.T.shape[1];
	KKT = numpy.zeros((KKT_rows, KKT_cols));

	#===Make KKT Matrix===#
	A1 = numpy.r_[numpy.ravel(A), numpy.zeros(KKT_cols - A.shape[1])];
	KKT = numpy.vstack((numpy.hstack((H, A.T)),A1));

	#===Make LHS===#
	lhs = numpy.matrix(numpy.r_[-numpy.ravel(grad_f), 0]).T;

	return KKT, lhs;

#===Run===#
m = 3.0;
tolerance = 10e-8;
xk = numpy.asmatrix(numpy.array([1.0/3.0,1.0/3.0,1.0/3.0])).T;
t = 1.0; u = 4.2;
for outer in range(5000):
	for inner in range(5000):

		#===Make as an array===#
		x = numpy.ravel(numpy.asarray(xk));
		x += eps;

		#===Make Gradient===#
		grad_f = numpy.asmatrix(numpy.array([x[0], x[1], 0.1*x[2] + 0.55])).T;
		grad_B = numpy.asmatrix(numpy.array([-1.0/(x[0]), -1.0/(x[1]), -1.0/(x[2])])).T;
		grad_tot = t*grad_f + grad_B;
		
		#===Make Hessian===#
		H_f = numpy.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0/10.0]]);
		H_B = numpy.matrix([[-(x[0])**-2.0, 0.0, 0.0], [0.0, -(x[1])**-2.0, 0.0], [0.0, 0.0, -(x[2])**-2.0]]);
		H_tot = t*H_f + H_B;

		#===Make Constraint Matrix===#
		A = numpy.matrix([1.0, 1.0, 1.0]);

		#===Make KKT Matrix==+#
		KKT, lhs = create_KKT_system(H_tot, A, grad_tot);

		#===Solve===#
		sol = numpy.linalg.solve(KKT, lhs);

		#===Make the Direction===#
		dk = sol[0:3];

		#===Do Backtracking===#
		alpha = find_alpha(xk, dk, t, 1000);

		#===Update===#
		last_xk = numpy.copy(xk);
		xk = xk + alpha*dk;
		print "Outer: ", outer, " Inner: ", inner, " Xk: ", xk.T, " Error: ", m/t;
		if (sum(numpy.abs(last_xk - xk))<tolerance):
			break;
	#===Check if Stop===#
	if (m/t < tolerance):
		break;
		quit();

	#===Update t===#
	t = u*t;
