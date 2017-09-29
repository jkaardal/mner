import numpy as np
import theano
import mner.optimizer
import mner.util.util
import mner.solvers.solvers
import mner.solvers.constraints
import mner.solvers.samplers
import sys

try:
    import matplotlib.pyplot as plt
    matplotlib_loaded = True
except ImportError:
    matplotlib_loaded = False

if theano.config.floatX == "float64":
    float_dtype = np.float64
elif theano.config.floatX == "float32":
    float_dtype = np.float32
else:
    float_dtype = np.float64

# demo number (choose 1-7)
demo_type = int(sys.argv[1])

# show plot at end
show_plot = True

# the current 'jack' jackknife of 'njack' jackknives
jack = 1
njack = 4

# fraction of the data reserved for training and cross-validation (remainder is testing)
train_fraction = 0.7
cv_fraction = 0.2

# generate correlated Gaussian noise images
ny = 16
nx = 16
nsamp = 100000;

ndim = ny*nx

def GP(x, y, x0, y0):
    Kinv = 1.0/3.0
    r = np.array([x-x0, y-y0]).reshape((2, 1))
    return np.exp(-np.dot(r.T, r)*Kinv)

C = np.zeros((ndim, ndim))
for i in range(ny):
    for j in range(nx):
        c = np.zeros((ny, nx))
        for k in range(ny):
            for l in range(nx):
                c[l, k] = GP(l, k, j, i);
        C[:, i*nx+j] = c.ravel()

C = np.dot(C.T, C)

[L, M, R] = np.linalg.svd(C)
X = np.dot(np.diag(np.sqrt(M)), R)

s = np.dot(np.random.randn(nsamp, ndim), X)

# show feature space covariance
#plt.imshow(np.dot(s.T, s), aspect='auto', interpolation='none')
#plt.show()

# feature scaling (highly recommended)
s, s_avg, s_std = mner.util.util.zscore_features(s)

# generate synthetic weights
def Gauss2D(xi, x0, sx, yi, y0, sy, A, phi):
    xm = np.tile(xi.reshape((1, xi.size)), (yi.size, 1))
    ym = np.tile(yi.reshape((yi.size, 1)), (1, xi.size))

    x = (xm - x0)*np.cos(phi) + (ym - y0)*np.sin(phi)
    y = (x0 - xm)*np.sin(phi) + (ym - y0)*np.cos(phi)

    return A*np.exp(-((x/sx)**2 + (y/sy)**2)/2.0)

xi = np.arange(1.0, 17.0, 1.0)
yi = np.arange(1.0, 17.0, 1.0)

F = np.zeros((ndim, 6))
F[:, 0] = np.reshape(Gauss2D(xi, 8.0, 1.5, yi, 11.0, 1.5, 1.0, 0.0), (ndim,))
F[:, 1] = np.reshape(Gauss2D(xi, 11.0, 1.5, yi, 11.0, 1.5, -0.5, 0.0), (ndim,))
F[:, 2] = np.reshape(Gauss2D(xi, 5.0, 1.5, yi, 11.0, 1.5, -0.5, 0.0), (ndim,))
F[:, 3] = np.reshape(Gauss2D(xi, 8.0, 1.5, yi, 8.0, 2.0, -0.5, 0.0), (ndim,))
F[:, 4] = np.reshape(Gauss2D(xi, 5.0, 1.5, yi, 8.0, 2.0, 0.3, -np.pi/4.0), (ndim,))
F[:, 5] = np.reshape(Gauss2D(xi, 11.0, 1.5, yi, 8.0, 2.0, 0.3, np.pi/4.0), (ndim,))

h = np.reshape(Gauss2D(xi, 8.0, 1.0, yi, 11.0, 1.0, 0.1, 0.0), (ndim, 1))

# show ground truth weights
#for i in range(6):
#    plt.subplot(1, 6, i+1)
#    plt.imshow(np.reshape(F[:, i], (ny, nx)), aspect='equal', interpolation='none')
#plt.show()

# generate responses of classifier
a = -3.5
g = 0.05
W = np.diag(np.array([1.0, -1.0, -1.0, -1.0, 1.0, 1.0]))
J = g*np.dot(F, np.dot(W, F.T))
x = a + np.dot(s, h).ravel() + np.sum(s * np.dot(s, J), axis=1)
p = 1.0/(1.0 + np.exp(-x))

y = np.zeros((nsamp,))
rnd = np.random.rand(nsamp,)
y[(rnd < p)] = 1.0

# test response probability (should be ~0.25)
#print np.mean(y)

# generate training, cross-validation, and test set Boolean indices
trainset, cvset, testset, nshift = mner.util.util.generate_dataset_logical_indices(train_fraction, cv_fraction, nsamp, njack)
trainset, cvset, testset = mner.util.util.roll_dataset_logical_indices(trainset, cvset, testset, nshift, jack-1)
datasets = {'trainset': trainset, 'cvset': cvset, 'testset': testset}

# model parameters
rank = 6
cetype = ["UV-linear-insert"]
rtype = ["nuclear-norm"]

# if J is symmetrized using linear constraints, need to set signs of eigenvalues
csigns = np.array([1, -1]*(rank/2))

# set scaling of cost function (for each data set)
fscale = {"trainset": -1, "cvset": -1, "testset": -1}

# choose solver
#solver = mner.solvers.solvers.IPMSolver
solver = mner.solvers.solvers.LBFGSSolver

# fit parameters (note the change for demo_type == 7 below)
factr = 1.0e10
lbfgs = 30

# set up the optimizer and solve
if demo_type == 1:
    # find a local minimum of the training set
    print "Demo 1: find a local minimum on the training set."
    print ""

    opt = mner.optimizer.Optimizer(y, s, rank, cetype=cetype, rtype=[], solver=solver, datasets=datasets, fscale=fscale, csigns=csigns, lbfgs=lbfgs, precompile=True, compute_hess=False, verbosity=2, iprint=1, factr=factr)

    # optimize the model
    x, ftrain = opt.optimize()

elif demo_type == 2:
    # attempt to find the global minimum of the training set
    print "Demo 2: attempt to find a global minimum on the training set (heuristic)."
    print ""

    global_solver = mner.solvers.solvers.MultiInitSearch

    opt = mner.optimizer.Optimizer(y, s, rank, cetype=cetype, rtype=[], solver=global_solver, datasets=datasets, fscale=fscale, csigns=csigns, lbfgs=lbfgs, precompile=True, compute_hess=False, verbosity=2, iprint=1, factr=factr, multi_init_search_solver=solver)

    # optimize the model
    x, ftrain = opt.optimize()

elif demo_type == 3:
    # one-dimensional regularization grid search
    print "Demo 3: one-dimensional grid search to find a minimum on the cross-validation set."
    print ""

    grid_solver = mner.solvers.solvers.GridSearch

    # set up the regularization grid
    hypergrid = np.arange(0.0, 0.021, 0.001)
    hypergrid = [np.tile(hypergrid.reshape((hypergrid.size, 1)), (1, rank))]

    opt = mner.optimizer.Optimizer(y, s, rank, cetype=cetype, rtype=rtype, solver=grid_solver, datasets=datasets, fscale=fscale, csigns=csigns, lbfgs=lbfgs, precompile=True, compute_hess=False, verbosity=2, iprint=1, factr=factr, grid_search_solver=solver, hypergrid=hypergrid)

    # optimize the model
    x, ftrain = opt.optimize()

elif demo_type == 4:
    # multi-dimensional regularization grid search
    print "Demo 4: multi-dimensional grid search to find a minimum on the cross-validation set."
    print ""

    grid_solver = mner.solvers.solvers.GridSearch

    # set up the regularization grid
    hypergrid = np.arange(0.0, 0.11, 0.05)
    hypergrid = [[hypergrid]*rank]

    # constrain the hypergrid to avoid redundancy
    cons = mner.solvers.constraints.SplitSign

    opt = mner.optimizer.Optimizer(y, s, rank, cetype=cetype, rtype=rtype, solver=grid_solver, datasets=datasets, fscale=fscale, csigns=csigns, lbfgs=lbfgs, precompile=True, compute_hess=False, verbosity=2, iprint=1, factr=factr, grid_search_solver=solver, hypergrid=hypergrid, grid_search_cons=cons)

    # optimize the model
    x, ftrain = opt.optimize()

elif demo_type == 5:
    # one-dimensional regularization Bayesian optimization
    print "Demo 5: one-dimensional Bayesian optimization to find a minimum on the cross-validation set."
    print ""

    bayes_solver = mner.solvers.solvers.BayesSearch

    # set up the regularization domain
    domain = [np.array([0.0, 1.0]*rank).reshape((rank, 2)).T]

    opt = mner.optimizer.Optimizer(y, s, rank, cetype=cetype, rtype=rtype, solver=bayes_solver, datasets=datasets, fscale=fscale, csigns=csigns, lbfgs=lbfgs, precompile=True, compute_hess=False, verbosity=2, iprint=1, factr=factr, bayes_search_solver=solver, bayes_search_domain=domain)

    # optimize the model
    x, ftrain = opt.optimize()

elif demo_type == 6:
    # multi-dimensional regularization Bayesian optimization
    print "Demo 6: multi-dimensional Bayesian optimization to find a minimum on the cross-validation set."
    print ""

    bayes_solver = mner.solvers.solvers.BayesSearch

    # set up the regularization domain
    domain = [[np.array([0.0, 1.0])]*rank]
    
    # constrain the domain to avoid redundancy
    cons = mner.solvers.constraints.SplitSign

    # customize the sampling function (Monte Carlo will fail for large rank otherwise!)
    sampler = mner.solvers.samplers.SplitSignSampling

    opt = mner.optimizer.Optimizer(y, s, rank, cetype=cetype, rtype=rtype, solver=bayes_solver, datasets=datasets, fscale=fscale, csigns=csigns, lbfgs=lbfgs, precompile=True, compute_hess=False, verbosity=2, iprint=1, factr=factr, bayes_search_solver=solver, bayes_search_domain=domain, bayes_search_cons=cons, bayes_search_sampler=sampler)

    # optimize the model
    x, ftrain = opt.optimize()

elif demo_type == 7:
    # multi-dimensional Bayesian optimization with annealing of the exploration weight
    print "Demo 7: multi-dimensional Bayesian optimization with annealing of the exploration weight."
    print ""

    bayes_solver = mner.solvers.solvers.BayesSearch

    # set up the regularization domain
    domain = [[np.array([0.0, 1.0])]*rank]

    # constrain the domain to avoid redundancy
    cons = mner.solvers.constraints.SplitSign

    # customize the sampling function (Monte Carlo will fail for large rank otherwise!)
    sampler = mner.solvers.samplers.SplitSignSampling

    # kernel variance and length scale
    kernel_variance = 2.0
    kernel_lengthscale = 0.1

    opt = mner.optimizer.Optimizer(y, s, rank, cetype=cetype, rtype=rtype, solver=bayes_solver, datasets=datasets, fscale=fscale, csigns=csigns, lbfgs=lbfgs, precompile=True, compute_hess=False, verbosity=2, iprint=1, factr=factr, bayes_search_solver=solver, bayes_search_domain=domain, bayes_search_cons=cons, bayes_search_sampler=sampler, bayes_search_kernel_variance_fixed=kernel_variance, bayes_search_kernel_lengthscale_fixed=kernel_lengthscale)

    # optimization settings (maxits, exploration_weight)
    settings = [(50, 2.0), (50, 1.0), (50, 0.0)] 

    x = opt.train_model.init_vec()
    for i in range(len(settings)):
        opt.solver.maxits = settings[i][0]
        opt.solver.exploration_weight = settings[i][1]

        x, ftrain = opt.optimize(x0=x)

elif demo_type == 8:
    # find solution in the globally optimal domain with minimal nuclear-norm regularization
    print "Demo 8: globally optimal approximation to the global minimum on the training set."
    print ""

    # greater precision is required for convergence
    factr = 1.0e7
    
    global_solver = mner.solvers.solvers.NNGlobalSearch

    opt = mner.optimizer.Optimizer(y, s, rank, cetype=cetype, rtype=rtype, solver=global_solver, datasets=datasets, fscale=fscale, csigns=csigns, lbfgs=lbfgs, precompile=True, compute_hess=False, verbosity=2, iprint=1, factr=factr, nn_global_search_solver=solver, nn_global_search_maxiter=100)

    # optimize the model
    x, ftrain = opt.optimize()

print "final ftrain = " + str(ftrain)
print "final fcv = " + str(opt.compute_set("cv"))
print "final test = " + str(opt.compute_set("test"))

# convert weight vector to coefficients
a, h, U, V = mner.util.util.vec_to_weights(x, ndim, rank)
V = np.dot(U, np.diag(csigns))

# form J and symmetrize then compute components
Jsym = np.dot(U, V.T)
Jsym = 0.5*(Jsym + Jsym.T)
[u, _, _] = np.linalg.svd(Jsym)

# compute the ground truth
[u_GT, _, _] = np.linalg.svd(0.5*(J+J.T))

if matplotlib_loaded and show_plot:
    for i in range(6):
        plt.subplot(2, 6, i+1)
        cm = np.max(np.abs(u_GT[:,i]))
        plt.imshow(np.reshape(u_GT[:,i], (ny, nx)), aspect='equal', interpolation='none', clim=(-cm, cm))
    for i in range(6):
        plt.subplot(2, 6, i+7)
        cm = np.max(np.abs(u[:,i]))
        plt.imshow(np.reshape(u[:,i], (ny, nx)), aspect='equal', interpolation='none', clim=(-cm, cm))
    plt.show()
