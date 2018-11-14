from cvxpy import *
import numpy

def form_gradient(x):
	x = numpy.ravel(x);
	grad = numpy.zeros(len(x));
	for n in range(len(x)):
		grad[n] = 1.0 + numpy.log(x[n]);
	grad = numpy.matrix(grad).T;
	return grad;


#===Setup===#
max_iters = 1000000;
alpha = 0.1;
beta = 0.75;
restol = 1e-7;

#====Create A===#
N = 3; P = 2;

#===Set >=0 ===#
xk =-1.0*numpy.ones(N);
while (min(xk) < 0):
	A = numpy.matrix(numpy.random.uniform(0.1, 1.0, (P,N)));
	b = numpy.matrix(numpy.random.uniform(0.1, 1.0, (P,1)));

	#===Solve and then make infeasible===#
	xk = numpy.linalg.lstsq(A,b)[0] + numpy.matrix(numpy.random.uniform(0.1, 1.0, (N,1)));
	vk = numpy.matrix(numpy.zeros(P)).T

#===Run Algorithm===#
r_norm = 1000.0;
iterations = 0;
while(1):

	xk1 = numpy.ravel(numpy.asarray(xk));

	#===Form Gradient===#
	grad_f = form_gradient(xk1);

	#===Form Hessian===#
	h = numpy.zeros(N);
	for n in range(N):
		h[n] = 1.0/xk1[n];
	H = numpy.matrix(numpy.diag(h));

	#===Form KKT Matrix===#
	KKT_rows = H.shape[0] + A.shape[0];
	KKT_cols = H.shape[1] + A.T.shape[1];
	KKT = numpy.zeros((KKT_rows, KKT_cols));

	#===Make KKT Matrix===#
	A1 = numpy.hstack((A, numpy.zeros((P,P))));
	KKT = numpy.vstack((numpy.hstack((H, A.T)),A1));
	KKT *= -1.0;

	#===Form Residual===#
	r1 = grad_f + A.T*vk; 
	r2 = A*xk-b;
	r = numpy.r_[numpy.ravel(r1), numpy.ravel(r2)];
	rnorm_last = r_norm;
	r_norm = numpy.linalg.norm(r);

	#===Solve===#
	sol = numpy.linalg.solve(KKT, r);
	dx = numpy.matrix(sol[0:N]).T; 
	dv = numpy.matrix(sol[N:N+P]).T;

	#===Find Step===#
	t = 1.0;
	while(1):

		#===Take Step In dom(f)===#
		xkk = xk + t*dx;
		while(min(xkk)<0):
			t = beta*t;
			xkk = xk + t*dx; 
		vkk = vk + t*dv;

		#===Form New Gradient===#
		grad = form_gradient(xkk);

		#===Form New Residual===#
		r11 = grad + A.T*vkk; r21 = A*xkk-b;
		r1 = numpy.r_[numpy.ravel(r11), numpy.ravel(r21)];
		r1_norm = numpy.linalg.norm(r1);

		if (r1_norm > (1.0-alpha*t)*r_norm):
			t = beta*t;
		else:
			break;

	#===Update X and V===#
	xk = xk + t*dx;
	vk = vk + t*dv;
	iterations += 1;

	#===Break Out if Stablized===#
	#if (numpy.abs(r_norm - rnorm_last) < 1e-8):
	if (r_norm < 1e-8):
		break;
	print "Iteration: ", iterations, " Rnorm: ", r_norm

print "\n";
print "Xk: ", xk.T
print "Vk: ", vk.T;
print "Rnorm: ", r_norm
print "Total Iterations: ", iterations

