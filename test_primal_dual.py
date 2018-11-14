from cvxpy import *
import numpy


'''
	minimize 0.5*x1**2 + 0.5*x2**2 + 0.05*x3**2 + 0.55x3
	subject to [1.0 1.0 1.0][x1] = 1.0
					  		[x2]
					  		[x3]
				-x1 <= 0
				-x2 <= 0
				-x3 <= 0
'''

def fi(x, i):
	return -1.0*x[i];

def form_residual(xk, lk, vk, t):

	#===Get Dimensions===#
 	N = xk.shape[0]; MI = lk.shape[0]; ME = vk.shape[0];

	#===Set Up A===#
	A = numpy.asmatrix([1.0, 1.0, 1.0]);
	b = numpy.ones((1,1));

	#===Make Gradient===#
	grad_f_temp = numpy.asmatrix(numpy.array([xk[0], xk[1], 0.1*xk[2] + 0.55])).T;
	
	#===Make Constraint Vector===#
	vec_f_temp = numpy.asmatrix(numpy.array([-xk[0], -xk[1], -xk[2]])).T;

	#===Make Derivative Matrix===#
	del_vec_f_temp = numpy.asmatrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]);

	#===Form Lambda Matrix===#
	diag_lambda_temp = numpy.diag(numpy.ravel(lk));

	#===Form Residual===#
	r1 = grad_f_temp + del_vec_f_temp.T*lk + A.T*vk;
	r2 = -diag_lambda_temp*vec_f_temp - (1.0/t)*numpy.ones((MI,1));
	r3 = A*xk - b;
	r_temp = numpy.r_[r1, r2, r3];

	return r_temp;

def make_primal_dual_system(xk, lk, vk, A, b, grad_f, H_f, vec_f, del_vec_f, H_fi, t):
		
	#===Get Dimensions===#
 	N = xk.shape[0]; MI = lk.shape[0]; ME = vk.shape[0];

	#==============================================================================================#
	#=======================================Form RHS===============================================#
	#==============================================================================================#

	#===Form First Row===#
	H_1_1 = numpy.copy(H_f);
	for m in range(MI):
		lk_m = numpy.ravel(lk[m])[0];
		H_1_1 += lk_m * H_fi[m];
	H_1_2 = del_vec_f.T;
	H_1_3 = A.T;
	H1 = numpy.c_[H_1_1, H_1_2, H_1_3];	

	#===Form Second Row===#
	diag_lk = numpy.diag(numpy.ravel(lk));
	H_2_1 = -diag_lk * del_vec_f;
	diag_vec_f = -numpy.diag(numpy.ravel(vec_f));
	H_2_2 = -diag_vec_f;
	H_2_3 = numpy.asmatrix(numpy.zeros((MI,ME)));
	H2 = numpy.c_[H_2_1, H_2_2, H_2_3];	
	
	#===Form Third Row===#
	Zero1 = numpy.zeros((ME,MI));
	Zero2 = numpy.zeros((ME,ME));
	H3 = numpy.c_[A, Zero1, Zero2];

	#===Concat to Matrix===#
	H = numpy.r_[H1, H2, H3];

	#==============================================================================================#
	#=======================================Form LHS===============================================#
	#==============================================================================================#

	g1 = grad_f + del_vec_f.T*lk + A.T*vk;
	g2 = -diag_lk*vec_f - (1.0/t)*numpy.ones((MI,1));
	g3 = A*xk - b;
	g = numpy.r_[-g1,-g2,-g3];

	return H, g;


#===Run===#
MI = 3; ME = 1; N = 3; 
mu = 1.5; beta = 0.95; alpha = 0.45;
xk = numpy.asmatrix(numpy.array([1.0/2.0,1.0/3.0,1.0/7.0])).T;
lk = numpy.asmatrix(numpy.array([1.0/3.0,1.0/2.0,1.0/5.0])).T;
vk = numpy.asmatrix(numpy.array([1.0])).T;
tolerance_feasibility = 10e-8; tolerance_duality = 10e-8;
t = 1.0;
count = 0;
while(1):

	#===Make as an array===#
	x = numpy.ravel(numpy.asarray(xk));

	#===Set Up A===#
	A = numpy.asmatrix([1.0, 1.0, 1.0]);
	b = numpy.ones((1,1));

	#===Make Gradient===#
	grad_f = numpy.asmatrix(numpy.array([x[0], x[1], 0.1*x[2] + 0.55])).T;
	
	#===Make Hessian===#
	H_f = numpy.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0/10.0]]);

	#===Make Constraint Vector===#
	vec_f = numpy.asmatrix(numpy.array([-x[0], -x[1], -x[2]])).T;

	#===Make Derivative Matrix===#
	del_vec_f = numpy.asmatrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]);

	#===Make Constraint Hessians===#
	H_fi = [ numpy.matrix(numpy.zeros((N,N))) for i in range(MI) ];

	#===Compute Duality Gap===#
	eta = vec_f.T*lk;
	t = numpy.ravel(mu*float(MI)/eta)[0];	

	#===Make Primal Dual System===#
	print "Solving KKT System...";
	KKT, lhs = make_primal_dual_system(xk, lk, vk, A, b, grad_f, H_f, vec_f, del_vec_f, H_fi, t);
	sol = numpy.linalg.solve(KKT, lhs);

	#===Extract Steps===#
	del_x = sol[0:N];
	del_l = sol[N:N+MI];
	del_v = sol[-1];


	#==============================================================================================#
	#=====================================Line Search==============================================#
	#==============================================================================================#

	#===Ensure Next Lambda is Feasible===#
	print "Lambda Feasibility...";
	s_max = 1.0;
	temp = numpy.ravel(del_l);
	print temp
	print "Doesnt seem to be working! Why?!?";
	quit();
	for i in range(MI):
		if (temp[i] < 0):
			s_max = min(1.0, min(-numpy.ravel(lk)[i]/numpy.ravel(del_l)[i]));
	print s_max
	lambda_plus = lk + s_max*del_l;
	if (count > 0):
		print lambda_plus.T

	#===Ensure Inequalities Are Satisfied===#
	print "Inequality Feasibility...";
	s = 0.99*s_max;
	x_plus = xk + s*del_x;
	vec_f_plus = numpy.asmatrix(numpy.array([-x_plus[0], -x_plus[1], -x_plus[2]])).T;
	while (vec_f_plus[0] > 0 or vec_f_plus[1] > 0 or vec_f_plus[2] > 0): 
		s = beta*s;
		x_plus = xk + s*del_x;
		vec_f_plus = numpy.asmatrix(numpy.array([-x_plus[0], -x_plus[1], -x_plus[2]])).T;
	if (count > 0):
		print vec_f_plus.T

	#===============#
	#===Backtrack===#
	#===============#
	print "Backtracking...";
	v_plus = vk + s*del_v;
	current_residual = form_residual(xk, lk, vk, t);
	residual_plus = form_residual(x_plus, lambda_plus, v_plus, t);
	while (numpy.linalg.norm(residual_plus) > (1.0 - alpha*s)*numpy.linalg.norm(current_residual)):
		s = beta*s;
		x_plus = xk + s*del_x;
		lambda_plus = lk + s*del_l;
		v_plus = vk + s*del_v;
		residual_plus = form_residual(x_plus, lambda_plus, v_plus, t);
		if (numpy.abs(numpy.linalg.norm(residual_plus) - (1.0 - alpha*s)*numpy.linalg.norm(current_residual)) < 10e-16):
			break;
	print "Val: ", numpy.linalg.norm(residual_plus), (1.0 - alpha*s)*numpy.linalg.norm(current_residual), s;

	#===Take Step===#
	sk = s;
	xk = xk + sk*del_x;
	lk = lk + sk*del_l;
	vk = vk + sk*del_v;
	count += 1;

	print "Xk: ", xk.T, " Lk: ", lk.T, " Vk: ", vk.T, "Sk: ", sk;
	print "\n";
	

	#===Check===#
	stepped_residual = form_residual(xk, lk, vk, t);
	rdual = stepped_residual[0]; rpri = stepped_residual[2];
	if (numpy.linalg.norm(rdual) < tolerance_feasibility and numpy.linalg.norm(rpri) < tolerance_feasibility and eta < tolerance_duality):
		break; 
	
