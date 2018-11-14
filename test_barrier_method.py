from cvxpy import *
import numpy

def find_alpha(xk, dk, ek, num_times):

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
			current_value = 0.5*(xk_new[0]**2.0 + xk_new[1]**2.0 + 0.1*xk_new[2]**2.0) + 0.55*xk_new[2]; 
			current_value -= ek*(numpy.log(xk_new[0]) + numpy.log(xk_new[1]) + numpy.log(xk_new[2]));

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
			current_value = 0.5*(xk_new[0]**2.0 + xk_new[1]**2.0 + 0.1*xk_new[2]**2.0) + 0.55*xk_new[2]; 
			current_value -= ek*(numpy.log(xk_new[0]) + numpy.log(xk_new[1]) + numpy.log(xk_new[2]));
			if current_value < best_value:
				best_value = current_value;
				current_alpha = this_alpha;
	return current_alpha;


#===Run===#
xk = numpy.asmatrix(numpy.array([1.0/3.0,1.0/3.0,1.0/3.0])).T;
#sk = 0.1;
sk = 1.0;
last_xk = numpy.copy(xk);
e = numpy.linspace(100.0, 10e-6, 10e3);
tolerance = 1e-10;
for k in range(5000):

	#===Create Variable===#
	x = Variable(3);

	#===Create Gradient===#
	xkk = numpy.ravel(numpy.asarray(xk));
	gradient = numpy.asmatrix(numpy.array([xkk[0] - e[k]/xkk[0], xkk[1] - e[k]/xkk[1], 0.1*xkk[2] + 0.55 - e[k]/xkk[2]])).T;

	#===Create Hessian===#
	H = numpy.asmatrix([[1.0 + e[k]*xkk[0]**(-2.0), 0.0, 0.0], [0.0, 1.0 + e[k]*xkk[1]**(-2.0), 0.0], [0.0, 0.0, 1.0/10.0 + e[k]*xkk[2]**(-2.0)]]);

	#===Create Cost===#
	c = Parameter(3);
	c.value = gradient - (1.0/sk)*H*xk;

	#===Create Diagonal Matrix===#
	D = numpy.asmatrix((1.0/sk) * numpy.ones((3,3)));

	#===Create Weighted Hessian===#
	Hw = numpy.multiply(D, H);

	#===Create Constant===#
	g = Parameter(1);
	g.value = (1.0/(2.0*sk))*xk.T*H*xk - gradient.T*xk;

	#===Create a Vector===#
	a = Parameter(3)
	a.value = numpy.array([1.0, 1.0, 1.0]);

	#===Create Constraint(s)===#
	constraints = [(a.T*x == 1.0)];

	#===Create And Solve Projection Problem===#
	obj = Minimize(0.5*quad_form(x, Hw) + c.T*x + g);
	prob = Problem(obj, constraints);
	prob.solve();
	
	#===Create The Direction===#
	xopt = numpy.asmatrix(numpy.asarray(prob.variables()[0].value)[:,0]).T;
	dk = (xopt - xk);
	alpha = find_alpha(xk, dk, e[k], 1000);

	#===Update===#
	last_xk = numpy.copy(xk);
	xk = xk + alpha*dk;
	if (sum(numpy.abs(last_xk - xk))<tolerance):
		break;
	print "k: ", k, " Xk: ", xk.T;

