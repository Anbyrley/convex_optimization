from cvxpy import *
import numpy

def find_alpha(xk, dk, num_times):

	if (0):
		#===Bisection Search For Alpha===#
		times = 0;
		current_value = 100000.0;
		left_alpha = 0.0; right_alpha = 1.0;
		while (1):

			#===Test Current Value===#
			current_alpha = (left_alpha+right_alpha)/2.0;
			last_value = numpy.copy(current_value);

			#===Make Xk===#
			xk_new = xk + current_alpha*dk;
			xk_new = numpy.asarray(xk_new);			
			current_value = -2.0*numpy.log(xk_new[0]) - 3.0*numpy.log(xk_new[1]) - 3.0*numpy.log(xk_new[2]);		

			#===Check===#
			if (times >= num_times):
				break;
			else: #===Move===#		
				if (current_value < last_value):
					move = abs(right_alpha - current_alpha);
					right_alpha = current_alpha;
				elif (current_value > last_value):
					move = abs(current_alpha - left_alpha);
					left_alpha = current_alpha;

			#===Update Times===#
			times += 1;
	else:
		#===Line Search For Alpha===#
		alphas = numpy.linspace(0, 1, num_times);
		best_value = 100000000.0; current_alpha = 1.0;
		for this_alpha in alphas:
			xk_new = xk + this_alpha*dk;
			xk_new = numpy.asarray(xk_new);
			current_value = -2.0*numpy.log(xk_new[0]) - 3.0*numpy.log(xk_new[1]) - 3.0*numpy.log(xk_new[2]);		
			if current_value < best_value:
				best_value = current_value;
				current_alpha = this_alpha;
	return current_alpha;

def create_residual( grad_f, A, b, xk, v ):

	r_dual = grad_f + A.T*v;
	r_pri = A*xk - b;

	return r_dual, r_pri;

def create_KKT_system( H, A, grad_f, b, xk ):

	#===Make KKT Shape===#
	KKT_rows = H.shape[0] + A.shape[0];
	KKT_cols = H.shape[1] + A.T.shape[1];
	KKT = numpy.zeros((KKT_rows, KKT_cols));

	#===Make KKT Matrix===#
	A1 = numpy.r_[numpy.ravel(A), numpy.zeros(KKT_cols - A.shape[1])];
	KKT = numpy.vstack((numpy.hstack((H, A.T)),A1));

	#===Make LHS===#
	b1 = numpy.ravel(A*xk)[0];
	lhs = numpy.matrix(numpy.r_[-numpy.ravel(grad_f), b - b1]).T;

	return KKT, lhs;


#===Run===#
tolerance = 1e-8; b = 10.0; vk = 5.0;
xk = numpy.asmatrix(numpy.array([2.0,3.0,7.0])).T;
for k in range(5000):

	#===Make as an array===#
	x = numpy.ravel(numpy.asarray(xk));

	#===Make Gradient===#
	grad_f = numpy.matrix([-2.0/x[0], -3.0/x[1], -3.0/x[2]]).T;
	
	#===Make Hessian===#
	H = numpy.matrix([[2.0*x[0]**-2.0, 0.0, 0.0], [0.0, 3.0*x[1]**-2.0, 0.0], [0.0, 0.0, 3.0*x[2]**-2.0]]);

	#===Make Constraint Matrix===#
	A = numpy.matrix([1.0, 2.0, 2.0]);

	#===Make KKT Matrix==+#
	KKT, lhs = create_KKT_system(H, A, grad_f, b, xk);

	#===Solve===#
	sol = numpy.linalg.solve(KKT, lhs);

	#===Make the Direction===#
	dk = sol[0:3];
	w = sol[3];
	delta_vk = w - vk;

	#===Do Backtracking===#
	t = 1.0; alpha = 0.25; beta = 0.5;
	while(1):

		#===Make Residual===#
		rdual, rpri = create_residual(grad_f, A, b, xk, vk);
		rr = numpy.r_[ numpy.ravel(rdual), numpy.ravel(rpri)];
		residual_k = (1.0 - alpha*t) * numpy.sqrt(sum(rr**2.0));

		#===Update xk and vk===#
		xk1 = xk + t*dk;
		vk1 = vk + t*delta_vk;

		#===Make Residual===#
		rdual, rpri = create_residual(grad_f, A, b, xk1, vk1);
		rr = numpy.r_[ numpy.ravel(rdual), numpy.ravel(rpri)];
		residual_k1 = numpy.sqrt(sum(rr**2.0));

		#===Update===#
		if (residual_k1 > residual_k):
			t = beta * t;

	#===Update===#
	last_xk = numpy.copy(xk);
	xk = xk + t*dk;
	if (sum(numpy.abs(last_xk - xk))<tolerance):
		break;
	print "k: ", k, " Xk: ", xk.T;

