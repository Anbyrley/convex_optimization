from cvxpy import *
import numpy

'''
	minimize 0.5 * (x1^2 + x2^2 + x3^2) + 0.55x3
	subject to 		x1 + x2 + x3 = 1.0
					x1 >= 0
					x2 >= 0
					x3 >= 0
	At iteration xk:
		gradf_k = [xk1, xk2, 0.1*xk3 + 0.55]
		We solve:
			minimize gradf_k^T(x - xk)
			subject to a^T x = 1
					   x1 >= 0
					   x2 >= 0
					   x3 >= 0
			where:
				a^T = [1.0, 1.0, 1.0]

		We transform:
			minimize c^T x
			subject to a^Tx = 1
					   x >= 0
			where:
				c^T = gradf_k
'''

def find_alpha(xk, dk, num_times):

	if (1):
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
			current_value = 0.5*(xk_new[0]**2.0 + xk_new[1]**2.0 + 0.1*xk_new[2]**2.0) + 0.55*xk_new[2]; 

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
			current_value = 0.5*(xk_new[0]**2.0 + xk_new[1]**2.0 + 0.1*xk_new[2]**2.0) + 0.55*xk_new[2]; 
			if current_value < best_value:
				best_value = current_value;
				current_alpha = this_alpha;
	return current_alpha;


#===Run===#
xk = numpy.array([1.0/3.0,1.0/3.0,1.0/3.0]);
alpha = 0.01;
last_xk = numpy.copy(xk);
tolerance = 1e-3;
for k in range(5000):
	
	#===Create Variable===#
	x = Variable(3);

	#===Create Cost===#
	c = Parameter(3);
	c.value = numpy.array([xk[0], xk[1], 0.1*xk[2] + 0.55]);

	#===Create a Vector===#
	a = Parameter(3)
	a.value = numpy.array([1.0, 1.0, 1.0]);

	#===Make Constant===#
	g = -(xk[0]**2.0) - (xk[1]**2.0) - (0.1*xk[2] + 0.55)*xk[2]; 

	#===Create Constraint(s)===#
	constraints = [(a.T*x == 1.0), (x >= 0)];

	#===Create And Solve Problem===#
	obj = Minimize(c.T*x + g);
	prob = Problem(obj, constraints);
	prob.solve();

	#===Make Direction===#
	xopt = numpy.asarray(prob.variables()[0].value)[:,0];
	dk = (xopt - xk);

	#===Find Alpha===#
	alpha = find_alpha(xk, dk, 1000);

	#===Update===#
	last_xk = numpy.copy(xk);
	xk = xk + alpha*dk;
	if (sum(numpy.abs(last_xk - xk))<tolerance):
		break;
	print "k: ", k, " Xk: ", xk;

#===Print Solution===#
print "\n";
print "Optimal Xk: ", xk;

