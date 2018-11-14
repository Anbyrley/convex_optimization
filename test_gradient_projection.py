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
			current_value = xk_new[0]**2.0 + (xk_new[1] + 1.0)**2.0

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
			current_value = xk_new[0]**2.0 + (xk_new[1] + 1.0)**2.0
			if current_value < best_value:
				best_value = current_value;
				current_alpha = this_alpha;
	return current_alpha;


#===Run===#
xk = numpy.array([1.0/3.0,1.0/3.0]);
sk = 1.0;
last_xk = numpy.copy(xk);
tolerance = 1e-6;
for k in range(5000):


	#===Create Gradient===#
	gradient = numpy.array([2.0*xk[0], 2.0*xk[1]+2.0]);

	#===Take Step===#
	xk1 = xk - sk * gradient; 

	#===Project the Step===#
	if (xk1[0] <= -1.0):
		xk1[0] = -1.0;
	if (xk1[0] >= 1.0):
		xk1[0] = 1.0;
	if (xk1[1] <= 0.0):
		xk1[1] = 0.0;

	#===Create The Direction===#
	dk = (xk1 - xk);
	alpha = find_alpha(xk, dk, 1000);

	#===Update===#
	last_xk = numpy.copy(xk);
	xk = xk + alpha*dk;
	if (sum(numpy.abs(last_xk - xk))<tolerance):
		break;
	print "k: ", k, " Xk: ", xk;

