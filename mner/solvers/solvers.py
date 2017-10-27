import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import theano
from . import manager

""" solvers.py (module)

    The solvers.py module contains classes for solving the low-rank
    MNE minimization problem. The classes included here include both
    local and global optimization solvers that find minima on the
    training set data and hyperparameter search algorithms for
    emprically determining the hyperparameter settings (such as
    regularization parameters).

    To add more solvers, follow along to the structure of the other
    solvers that have been provided here.

"""

# solver classes
class BaseSolver(object):
    """ BaseSolver (class)

        Base solver class that provides the essential variables and class
        functions that may be needed by child solver classes.  Solver
        classes find either a local or global minimum of the low-rank
        MNE problem on the training set.

    """
    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the instantiation of class BaseSolver.

            [inputs] (parent=dict(), **kwargs) 
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class Optimizer
                  from module mner.optimizer.py) from which the solver
                  is instantiated.
                  - x0 (default=None): initial weights
                  - train_model (default=None): low-rank MNE model
                    evaluated on the training set
                  - float_dtype (default=np.float64): floating-point
                    data type
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - x0: same as above
                  - train_model: same as above
                  - float_dtype: same as above

        """
        # get weight initialization
        self.x0 = kwargs.get('x0', parent.get('x0', None))

        # get model
        self.train_model = kwargs.get('train_model', parent.get('train_model', None))

        # hold on to parent
        self.parent = parent

        # get floating-point data type
        self.float_dtype = kwargs.get('float_dtype', parent.get('float_dtype', np.float64))

        self.initialized = True

    def init_solver(self, **kwargs):
        """ If the solver itself needs to optimize a subproblem, this function
            initializes the subproblem solver.
            
            [inputs] (**kwargs)

        """
        # initialize a nested solver (e.g. for solving a subproblem)
        if not hasattr(self.solver, 'initialized') or self.solver.initialized == False:
            self.solver = self.solver(self, **kwargs)

    def get(self, name, default=None):
        """ Get attribute, if it exists; otherwise, return default.

            [inputs] (name, default=None)
                name: string identifying the attribute name.
                default: (optional) if attribute does not exist,
                return a default value.
        
            [returns] attr_val
                attr_val: either the requested attribute identified by
                name or the default, when appropriate.

        """
        return getattr(self, name, default)

    def __getitem__(self, name):
        return self.get(name)
    

            
class IPMSolver(BaseSolver):
    """ IPMSolver (class)

        Use an interior-point algorithm (from package pyipm) to locate a
        local minimum of the low-rank MNE problem subject to
        regularization and/or constraints.

    """
    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the interior-point solver including compiling Theano
            expressions into device functions (if that has not been
            done already).

            [inputs] (parent=dict(), **kwargs) 
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class Optimizer
                  from module mner.optimizer.py) from which the solver
                  is instantiated.
                  - parameters are inherited from BaseSolver
                  - x_dev: (default=None) Theano tensor for the
                    weights
                  - lda_dev: (default=None) Theano tensor for the
                    Lagrange multipliers
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - parameters are inherited from BaseSolver
                  - x_dev: same as above
                  - lda_dev: same as above
                  - for additional optional arguments, see
                    pyipm.py. These additional arguments may be set
                    using the same keyword arguments as found in
                    pyipm.py. To avoid ambiguity, 'ipm_solver_' can be
                    prepended to the keyword arguments as well;
                    e.g. can pass either mu or ipm_solver_mu as
                    keyword arguments.

        """
        # import solver class
        import pyipm

        # initialize base class
        super(IPMSolver, self).__init__(parent, **kwargs)
        
        # get variable initializations, if they exist
        self.lda0 = kwargs.get('ipm_solver_lda0', kwargs.get('lda0', None))
        self.s0 = kwargs.get('ipm_solver_s0', kwargs.get('s0', None))

        # get theano variables
        self.x_dev = kwargs.get('x_dev', parent.get('x_dev'))
        self.lambda_dev = kwargs.get('lda_dev', parent.get('lda_dev'))

        # get miscellaneous variables
        self.mu = kwargs.get('ipm_solver_mu', kwargs.get('mu', 0.2))
        self.nu = kwargs.get('ipm_solver_nu', kwargs.get('nu', 10.0))
        self.rho = kwargs.get('ipm_solver_rho', kwargs.get('rho', 0.1))
        self.tau = kwargs.get('ipm_solver_tau', kwargs.get('tau', 0.995))
        self.eta = kwargs.get('ipm_solver_eta', kwargs.get('eta', 1.0e-4))
        self.beta = kwargs.get('ipm_solver_beta', kwargs.get('beta', 0.4))
        self.miter = kwargs.get('ipm_solver_miter', kwargs.get('miter', 20))
        self.niter = kwargs.get('ipm_solver_niter', kwargs.get('niter', 10))
        self.Xtol = kwargs.get('ipm_solver_Xtol', kwargs.get('Xtol', None))
        self.Ktol = kwargs.get('ipm_solver_Ktol', kwargs.get('Ktol', 1.0E-4))
        self.Ftol = kwargs.get('ipm_solver_Ftol', kwargs.get('Ftol', None))
        self.lbfgs = kwargs.get('ipm_solver_lbfgs', kwargs.get('lbfgs', False))
        self.lbfgs_zeta = kwargs.get('ipm_solver_lbfgs_zeta', kwargs.get('lbfgs_zeta', None))
        self.verbosity = kwargs.get('ipm_solver_verbosity', kwargs.get('verbosity', 1))
        self.precompile = kwargs.get('ipm_solver_precompile', kwargs.get('precompile', 1))
        
        # get functions
        if self.precompile:
            f = self.train_model.cost
            df = self.train_model.grad
            d2f = self.train_model.hess
            ce = self.train_model.ceq
            dce = self.train_model.ceq_jaco
            d2ce = self.train_model.ceq_hess
            ci = self.train_model.cineq
            dci = self.train_model.cineq_jaco
            d2ci = self.train_model.cineq_hess
        else:
            f = self.train_model.f
            df = self.train_model.df
            d2f = self.train_model.d2f
            ce = self.train_model.ce
            dce = self.train_model.dce
            d2ce = self.train_model.d2ce
            ci = self.train_model.ci
            dci = self.train_model.dci
            d2ci = self.train_model.d2ci
            
        # initialize IPM
        self.problem = pyipm.IPM(x0=self.x0, x_dev=self.x_dev, f=f, df=df, d2f=d2f, ce=ce, dce=dce, d2ce=d2ce, ci=ci, dci=dci, d2ci=d2ci, lda0=self.lda0, lambda_dev=self.lambda_dev, s0=self.s0, mu=self.mu, nu=self.nu, rho=self.rho, tau=self.tau, eta=self.eta, beta=self.beta, miter=self.miter, niter=self.niter, Xtol=self.Xtol, Ktol=self.Ktol, Ftol=self.Ftol, lbfgs=self.lbfgs, lbfgs_zeta=self.lbfgs_zeta, float_dtype=self.float_dtype, verbosity=self.verbosity)
        # compilation will occur on the first call of self.solve()
        

    def solve(self, x0=None, **kwargs):
        """ Find a feasible local minimum using an interior-point algorithm.

            [inputs] (x0, **kwargs)
                x0: numpy array of the weight initialization (default
                  is set to the class member self.x0)

            [returns] (x, ftrain)
                x: numpy array of the weights that most closely
                  minimize the objective function.
                ftrain: scalar value of the objective function
                  evaluated at x.

        """
        if x0 is None:
            x0 = self.x0
        self.x, self.s, self.lda, self.ftrain, self.kkt = self.problem.solve(x0=x0)
        return (self.x.astype(self.float_dtype), self.ftrain.astype(self.float_dtype))






class LBFGSSolver(BaseSolver):
    """ LBFGSSolver (class)

        Find a feasible local minimizer of the low-rank MNE problem using
        bound constrained Limited-memory Broyden-Fletcher-
        Goldfarb-Shanno algorithm (L-BFGS) fmin_l_bfgs_b from scipy
        subpackage scipy.optimize.

        Note that this only accepts unconstrained, bound constrained,
        and/or constraints that can be directly substituted into the
        objective function. This likely rules out most nonlinear
        constraints and inequality constraints.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the L-BFGS solver instantiation.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class Optimizer
                  from module mner.optimizer.py) from which the solver
                  is instantiated.
                  - parameters are inherited from BaseSolver
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - parameters are inherited from BaseSolver
                  - (lbfgs_solver_)lbfgs: (default=20) number of
                    iterations to store for the Hessian approximation.
                    Note that this complete replaces m in the
                    scipy.optimize.fmin_l_bfgs_b docs!
                  - for additional optional arguments, see
                    documentation for
                    scipy.optimize.fmin_l_bfgs_b. These additional
                    arguments may be set using the same keyword
                    arguments as found in fmin_l_bfgs_b. To avoid
                    ambiguity, 'lbfgs_solver_' can be prepended to the
                    keyword arguments as well; e.g. can pass either
                    factr or lbfgs_solver_factr as keyword arguments.

        """
        # initialize base class
        super(LBFGSSolver, self).__init__(parent, **kwargs)
        
        # solver-specific options
        self.bounds = kwargs.get('lbfgs_solver_bounds', kwargs.get('bounds', None))
        self.lbfgs = kwargs.get('lbfgs_solver_lbfgs', kwargs.get('lbfgs', 20))
        self.factr = kwargs.get('lbfgs_solver_factr', kwargs.get('factr', 1.0e7))
        self.pgtol = kwargs.get('lbfgs_solver_pgtol', kwargs.get('pgtol', 1.0e-5))
        self.epsilon = kwargs.get('lbfgs_solver_epsilon', kwargs.get('epsilon', 1.0e-8))
        self.iprint = kwargs.get('lbfgs_solver_iprint', kwargs.get('iprint', 0))
        self.disp = kwargs.get('lbfgs_solver_disp', kwargs.get('disp', None))
        self.maxfun = kwargs.get('lbfgs_solver_maxfun', kwargs.get('maxfun', 15000))
        self.maxiter = kwargs.get('lbfgs_solver_maxiter', kwargs.get('maxiter', 15000))
        self.maxls = kwargs.get('lbfgs_solver_maxls', kwargs.get('maxls', 20))

        self.initialized = True


    def solve(self, x0=None, **kwargs):
        """ Find a feasible local minimum using an interior-point algorithm.

            [inputs] (x0=None, **kwargs)
                x0: (optional) numpy array of the weight
                  initialization (default is set to the class member
                  self.x0)

            [returns] (x, ftrain)
                x: numpy array of the weights that most closely
                  minimize the objective function.
                ftrain: scalar value of the objective function
                  evaluated at x.

        """
        if x0 is None:
            x0 = self.x0
        self.x, self.ftrain, _ = fmin_l_bfgs_b(func=lambda x: self.train_model.cost(x.astype(self.float_dtype)).astype(np.float64), x0=x0, fprime=lambda x: self.train_model.grad(x.astype(self.float_dtype)).astype(np.float64), bounds=self.bounds, m=self.lbfgs, factr=self.factr, pgtol=self.pgtol, epsilon=self.epsilon, iprint=self.iprint, disp=self.disp, maxfun=self.maxfun, maxiter=self.maxiter, maxls=self.maxls)
        return (self.x.astype(self.float_dtype), self.ftrain.astype(self.float_dtype))
        





        
# regularization solvers
class BaseSearch(object):
    """ BaseSearch (class)

        Base search class that provides the essential variables and class
        functions that may be needed by child search classes.  Search
        classes attempt to minimize the objective function evaluated
        on the cross-validation set with respect to hyperparameters
        such as regularization parameters.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the base search class instantiation.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class Optimizer
                  from module mner.optimizer.py) from which the solver
                  is instantiated.
                  - train_model: (default=None) low-rank MNE model
                    evaluated on the training set
                  - cv_model: (default=None) low-rank MNE model
                    evaluated on the cross-validation set
                  - float_dtype: (default=np.float64) floating-point
                    data type
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - max_storage: (default None) maximum number of
                    solutions to store. When set to None, the
                    max_storage is infinite.
                  - train_model: same as above
                  - cv_model: same as above
                  - float_dtype: same as above

        """
        # get training and cross-validation models
        self.train_model = kwargs.get('train_model', parent.get('train_model', None))
        self.cv_model = kwargs.get('cv_model', parent.get('cv_model', None))

        # initialize storage parameters
        self.max_storage = kwargs.get('max_storage', None)
        self.storage_initialized = False

        # get floating-point data type
        self.float_dtype = kwargs.get('float_dtype', parent.get('float_dtype', np.float64))

        # hold on to the parent
        self.parent = parent

        self.initialized = True
        

    def init_storage(self, hyperparams, **kwargs):
        """ Initialize/reinitialize empty storage arrays to store weights,
            hyperparameters, and objective function evaluations on the
            cross-validation set.

            [inputs] (hyperparams, **kwargs)
                hyperparams: vector of the (full-sized) hyperparameters
                  (see mner.solvers.manager.py)

        """
        self.x_storage = np.zeros((0, self.train_model.nvar))
        self.hyperparam_storage = np.zeros((0, hyperparams.size))
        self.hyperparam_norm_storage = np.zeros((0, 1))
        self.fcv_storage = np.zeros((0, 1))

        self.storage_initialized = True


    def update_storage(self, x, hyperparams, fcv, **kwargs):
        """ Update the storage arrays.

            [inputs] (x, hyperparams, fcv, **kwargs)
                x: weight vector
                hyperparams: vector of the (full-sized) hyperparameters
                  (see mner.solvers.manager.py)
                fcv: scalar value of the objective function evaluated
                  on the cross-validation set

        """
        self.x_storage = np.concatenate([self.x_storage, x.reshape((1, x.size))], axis=0)
        self.hyperparam_storage = np.concatenate([self.hyperparam_storage, hyperparams.reshape((1, hyperparams.size))], axis=0)
        self.hyperparam_norm_storage = np.concatenate([self.hyperparam_norm_storage, np.linalg.norm(hyperparams).reshape((1,1))], axis=0)
        self.fcv_storage = np.concatenate([self.fcv_storage, np.array([fcv]).reshape((1,1))], axis=0)

        if self.max_storage is not None and self.fcv_storage.size > self.max_storage:
            index = np.argmax(self.fcv_storage)
            self.x_storage = np.delete(self.x_storage, index, axis=0)
            self.hyperparam_storage = np.delete(self.hyperparam_storage, index, axis=0)
            self.hyperparam_norm_storage = np.delete(self.hyperparam_norm_storage, index, axis=0)
            self.fcv_storage = np.delete(self.fcv_storage, index, axis=0)


    def clear_storage(self, **kwargs):
        """ Clear the storage arrays and set to size zero.

            [inputs] (**kwargs)

        """
        self.x_storage = np.array([])
        self.hyperparam_storage = np.array([])
        self.hyperparam_norm_storage = np.array([])
        self.fcv_storage = np.array([])
        
        self.storage_initialized = False


    def init_weights_to_closest(self, parent=dict(), **kwargs):
        """ Initialize weights to the solution of the closest set of
            hyperparameters.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class Optimizer
                  from module mner.optimizer.py) from which the solver
                  is instantiated.
                  - hyperparams: (default=None) vector of
                    hyperparameters in the full hyperparameter space
                    (see mner.solvers.manager.py)
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - hyperparams: same as above
                  - round_type: (default="floor") string indicating
                    whether to round down ('floor'), round up
                    ('ceil'), or round to the closest ('round') set of
                    hyperparameters when choosing the
                    initialization. Rounding up or down is determined
                    by the norm of the hyperparameter vectors. The
                    default is 'floor'.

            [returns] x0
                x0: numpy array of a weight initialization.

        """
        hyperparams = kwargs.get('hyperparams', parent.get('hyperparams', None))
        round_type = kwargs.get('round_type', "floor")
        if self.storage_initialized and self.fcv_storage.size > 0:
            if round_type.strip().lower() == "floor":
                hyperparam_norm = np.linalg.norm(hyperparams)
                index = np.where(self.hyperparam_norm_storage <= hyperparam_norm)[0]
                if len(index) == 0:
                    index = np.where(self.hyperparam_norm_storage > hyperparam_norm)[0]
                r = np.linalg.norm(self.hyperparam_storage[index,:] - np.tile(hyperparams.reshape((1, hyperparams.size)), (len(index), 1)), axis=1)
                index2 = np.argmin(r)
                return np.copy(self.x_storage[index[index2],:].ravel())
            elif round_type.strip().lower() == "ceil":
                hyperparam_norm = np.linalg.norm(hyperparams)
                index = np.where(self.hyperparam_norm_storage >= hyperparam_norm)[0]
                if len(index) == 0:
                    index = np.where(self.hyperparam_norm_storage < hyperparam_norm)[0]
                r = np.linalg.norm(self.hyperparam_storage[index,:] - np.tile(hyperparams.reshape((1, hyperparams.size)), (len(index), 1)), axis=1)
                index2 = np.argmin(r)
                return np.copy(self.x_storage[index[index2],:].ravel())
            else:
                r = np.linalg.norm(self.hyperparam_storage - np.tile(hyperparams.reshape((1, hyperparams.size)), (self.hyperparam_norm_storage.shape[0], 1)), axis=1)
                index = np.argmin(r)
                return np.copy(self.x_storage[index,:].ravel())
        else:
            return np.copy(self.x)


    def init_solver(self, **kwargs):
        """ If the search needs to optimize a subproblem, this function
            initializes the subproblem solver.
            
            [inputs] (**kwargs)

        """
        if not hasattr(self.solver, 'initialized') or self.solver.initialized == False:
            self.solver = self.solver(self, **kwargs)

    def get(self, name, default=None):
        """ Get attribute, if it exists; otherwise, return default.

            [inputs] (name, default=None)
                name: string identifying the attribute name.
                default: (optional) if attribute does not exist,
                return a default value.
        
            [returns] attr_val
                attr_val: either the requested attribute identified by
                name or the default, when appropriate.

        """
        return getattr(self, name, default)

    def __getitem__(self, name):
        return self.get(name)


class GridSearch(BaseSearch):
    """ GridSearch (class)

        Perform a grid search to determine the hyperparameter settings.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the grid search.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class Optimizer
                  from module mner.optimizer.py) from which the solver
                  is instantiated.
                  - parameters are inherited from BaseSearch
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - parameters are inherited from BaseSearch
                  - (grid_search_)solver: (default=None) subproblem
                    solver for optimizing the training set objective
                    function
                  - (grid_search_)forget: (default=False) when True
                    performs a memoryless optimization where past
                    iterations are not stored
                  - (grid_search_)init_x:
                    (default=GridSearch.init_weights_to_closest)
                    weight initialization function
                  - (grid_search_)verbosity: (default=1) amount of
                    information printed to screen during the
                    optimization. Options are integers between -1 and
                    4.

        """
        # initialize base class
        super(GridSearch, self).__init__(parent, **kwargs)

        # set-up regularization manager
        self.hyper_manager = manager.HyperManager('grid_search', parent, **kwargs)

        # get and initialize subproblem solver
        self.solver = kwargs.get('grid_search_solver', kwargs.get('solver', None))
        if self.solver is not None:
            self.init_solver(**kwargs)

        # set whether to remember storage or not
        self.forget = kwargs.get('grid_search_forget', kwargs.get('forget', False))

        # initialization function
        self.init_x = kwargs.get('grid_search_init_x', kwargs.get('init_x', self.init_weights_to_closest))

        # verbosity level
        self.verbosity = kwargs.get('grid_search_verbosity', kwargs.get('verbosity', 1))

        self.complete = False


    def solve(self, x0=None, **kwargs):
        """ Solve for the hyperparameter settings that minimize the
            objective/cost function on the cross-validation set.

            [inputs] (x0=None, **kwargs)
                x0: (optional) numpy array of the weight
                  initialization (default is set to the class member
                  self.x0)
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - (grid_search_)hypergrid_index: (default=None)
                    integer numpy array of hypergrid indices where
                    each element correponds to an element of the
                    reduced hyperparameter space.

            [returns] (x, ftrain)
                x: numpy array of the weights that most closely
                  minimize the objective function.
                ftrain: scalar value of the objective function
                  evaluated at x.

        """
        # initialize solution vector and costs
        self.x = np.copy(x0.astype(self.float_dtype))
        self.ftrain = None
        self.fcv = np.inf

        # if a starting point in the grid is not provided, start from the beginning
        self.hypergrid_index = kwargs.get('grid_search_hypergrid_index', kwargs.get('hypergrid_index', None))
        if self.hypergrid_index is None:
            self.hypergrid_index = np.zeros((self.hyper_manager.red_index[-1],), dtype=np.uint32)

        if self.verbosity >= 0:
            print "Starting grid search..."

        # begin optimization
        while not self.complete:
            # get the new values on the grid and assign to regularization parameters
            self.hyper_manager.build_state_from_grid(self.hypergrid_index, **kwargs)
            self.hyper_manager.update_model(**kwargs)
            self.hyperparams = self.hyper_manager.build_vector_from_state(**kwargs)

            # initialize storage, if applicable; initialize x
            if not self.forget and not self.storage_initialized:
                self.init_storage(self.hyperparams, **kwargs)
            x0 = np.copy(self.init_x(self, **kwargs))

            # check feasibility of regularization parameters
            feasible = self.hyper_manager.check_feasibility(**kwargs)

            if self.verbosity >= 2:
                print "hyperparameters:"
                print self.hyper_manager.build_red_vector_from_state(**kwargs)
                if feasible:
                    print "hyperparameters are feasible."
                else:
                    print "hyperparameters are infeasible."

            #print self.hyperparams
            if feasible:                
                self.xtmp, self.ftmp = self.solver.solve(x0, **kwargs)
                self.fnew = self.cv_model.cost(self.xtmp)

                if self.fnew < self.fcv:
                    self.fcv = np.copy(self.fnew)
                    self.ftrain = np.copy(self.ftmp)
                    self.x = np.copy(self.xtmp)
                
                if self.verbosity >= 1:
                    print "fnew = " + str(self.fnew) + ", fcv (best) = " + str(self.fcv)

                if self.verbosity >= 3:
                    print "weights:"
                    print self.xtmp
                
                # update storage, if applicable
                if not self.forget:
                    self.update_storage(self.x, self.hyperparams, self.fnew, **kwargs)

            k_control = 0
            self.hypergrid_index[k_control] += 1
            while self.hypergrid_index[k_control] >= self.hyper_manager.length[k_control]:
                self.hypergrid_index[k_control] = 0
                k_control += 1
                if k_control < self.hyper_manager.red_index[-1]:
                    self.hypergrid_index[k_control] += 1
                else:
                    self.complete = True
                    break

            if self.verbosity >= 2:
                print ""

        if self.verbosity >= 0:
            if self.complete:
                print "Grid search complete."

        return (self.x.astype(self.float_dtype), self.ftrain.astype(self.float_dtype))

        
        


class BayesSearch(BaseSearch):
    """ BayesSearch (class)

        Search for the hyperparameter settings using Bayesian
        optimization.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the Bayesian optimization search.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class Optimizer
                  from module mner.optimizer.py) from which the solver
                  is instantiated.
                  - parameters are inherited from BaseSearch
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - parameters are inherited from BaseSearch
                  - (bayes_search_)solver: (default=None) subproblem
                    solver for optimizing the training set objective
                    function
                  - (bayes_search_)forget: (default=False) when True
                    performs a memoryless optimization where past
                    iterations are not stored
                  - (bayes_search_)init_x:
                    (default=GridSearch.init_weights_to_closest)
                    weight initialization function
                  - (bayes_search_)N_init: (default=4) positive scalar
                    dictating the number of initial points to compute
                    to initialize the Gaussian process in the Bayesian
                    optimization.
                  - (bayes_search_)maxits: (default=1000) positive
                    integer setting the maximum number of acquisitions
                    (iterations).
                  - (bayes_search_)epsilon: (default=1.0e-4) scalar
                    number setting the minimum norm difference between
                    consecutive hyperparameter acquisitions
                    convergence condition
                  - (bayes_search_)exact_feval: (default=True) Boolean
                    indicating if the objective function value is
                    deterministic (not probabilistic) for a given set
                    of hyperparameters
                  - (bayes_search_)normalize_Y: (default=True) Boolean
                    indicating whether to normalize the objective
                    function values
                  - (bayes_search_)acquisition_type: (default="LCB")
                    string to choose the type of acquisition
                    function. For more options see the GPyOpt
                    documentation.
                  - (bayes_search_)exploration_weight: (default=2.0)
                    positive scalar value that dictates the balance
                    between exploration and exploitation (convergence)
                    where larger numbers explore while number closer
                    to zero explore less. This is only relevant to
                    some acquisition functions like 'LCB'. For more
                    information, see the GPyOpt documentation.
                  - (bayes_search_)kernel: (default=GPy.kern.matern32)
                    covariance kernel used in the Gaussian
                    process. For more options, see the GPyOpt
                    documentation.
                  - (bayes_search_)kernel_variance_fixed:
                    (default=None; the variance coefficient is
                    optimized) hold the variance coefficient of the
                    Gaussian process's covariance kernel fixed to a
                    scalar value.
                  - (bayes_search_)kernel_lengthscale_fixed:
                    (default=None; the length scale parameter is
                    optimized) hold the length scale parameter of the
                    Gaussian process's covariance kernel fixed to a
                    scalar value
                  - (bayes_search_)acquisition_jitter:
                    (default=np.finfo(kwargs['float_dtype']).eps) the
                    scalar magnitude of white noise covariance added
                    to the covariance kernel of the Gaussian process
                  - (bayes_search_)model_type: (default='GP') string
                    dictating the model used for the likelihood
                    function. For more options, see the GPyOpt
                    documentation.
                  - (bayes_search_)model_update_interval: (default=1)
                    positive integer defining the number acquisitions
                    between updating the parameters of the covariance
                    kernel, if applicable.
                  - (bayes_search_)verbosity: (default=1) amount of
                    information printed to screen during the
                    optimization. Options are integers between -1 and
                    4.

        """
        # import the Bayesian optimization package
        import GPyOpt

        # initialize base class
        super(BayesSearch, self).__init__(parent, **kwargs)

        # set up the regularization manager
        self.hyper_manager = manager.HyperManager('bayes_search', parent, **kwargs)
        self.hyper_manager.format_domain(**kwargs)
        self.hyper_manager.format_cons(**kwargs)
        
        # get optimization parameters
        self.maxits = kwargs.get('bayes_search_maxits', kwargs.get('maxits', 1000))
        self.epsilon = kwargs.get('bayes_search_epsilon', kwargs.get('epsilon', 1.0e-4))
        self.exact_feval = kwargs.get('bayes_search_exact_feval', kwargs.get('exact_feval', True))
        self.normalize_Y = kwargs.get('bayes_search_normalize_Y', kwargs.get('normalize_Y', True))

        # acquisition function type and exploration weight parameter, if applicable
        self.acquisition_type = kwargs.get('bayes_search_acquisition_type', kwargs.get('acquisition_type', 'LCB'))
        self.exploration_weight = kwargs.get('bayes_search_exploration_weight', kwargs.get('exploration_weight', 2.0))

        # kernel function
        self.kernel = kwargs.get('bayes_search_kernel', kwargs.get('kernel', None))
        self.kernel_variance_fixed = kwargs.get('bayes_search_kernel_variance_fixed', kwargs.get('kernel_variance_fixed', None))
        self.kernel_lengthscale_fixed = kwargs.get('bayes_search_kernel_lengthscale_fixed', kwargs.get('kernel_lengthscale_fixed', None))

        # "white noise" added to Gaussian kernel (scaled identity matrix)
        self.acquisition_jitter = kwargs.get('bayes_search_acquisition_jitter', kwargs.get('acquisition_jitter', np.sqrt(np.finfo(parent.get('float_dtype', np.float64)).eps)))

        # Bayesian model type (keep this as a Gaussian process 'GP' under most circumstances)
        self.model_type = kwargs.get('bayes_search_model_type', kwargs.get('model_type', 'GP'))

        # how many iterations to skip between updates of the Gaussian process parameters
        self.model_update_interval = kwargs.get('bayes_search_model_update_interval', kwargs.get('model_update_interval', 1))

        # get and initialize solver
        self.solver = kwargs.get('bayes_search_solver', kwargs.get('solver', None))
        if self.solver is not None:
            self.init_solver(**kwargs)

        # forget storage history
        self.forget = kwargs.get('bayes_search_forget', kwargs.get('forget', True))

        # initialization function
        self.init_x = kwargs.get('bayes_search_init_x', kwargs.get('init_x', self.init_weights_to_closest))

        # initial number of regularization configurations to try
        self.N_init = kwargs.get('bayes_search_N_init', kwargs.get('N_init', 4))

        # verbosity level
        self.verbosity = kwargs.get('bayes_search_verbosity', kwargs.get('verbosity', 1))

        self.converged = False
        self.resume = False


    def init_points(self, N=4, **kwargs):
        """ Choose points on the domain of the hyperparameters to initialize
            the Bayesian optimization (hyperparameters are assumed to
            be continuous at the moment).

            [inputs] (N=4, **kwargs)
 
                N: (optional) non-negative integer dictating the
                  number of additional points to try.

            [returns] hyperparams_red_init
                hyperparams_red_init: numpy array of an initialization
                  for the reduced hyperparameters (see
                  mner.solvers.manager.py).

        """
        lower_bound = np.zeros((self.hyper_manager.red_index[-1],))
        upper_bound = np.zeros((self.hyper_manager.red_index[-1],))
        for i in range(lower_bound.size):
            lower_bound[i] = self.hyper_manager.domain_f[i]['domain'][0]
            upper_bound[i] = self.hyper_manager.domain_f[i]['domain'][-1]
            
        hstep = np.zeros((lower_bound.size,))
        for i in range(hstep.size):
            hstep[i] = np.log(1.0 + upper_bound[i])/N
        hyperparams_red_init = np.zeros((N+1, hstep.size))
        for i in range(hstep.size):
            for j in range(N+1):
                hyperparams_red_init[j, i] = lower_bound[i] + np.exp(hstep[i]*j) - 1.0
        hyperparams_red_init = hyperparams_red_init[~np.isnan(hyperparams_red_init).any(axis=1)]

        return hyperparams_red_init.astype(self.float_dtype)


    def hidden_cost(self, hyperparams_red, **kwargs):
        """ Compute the value of the "hidden" objective/cost function. The
            training model is solved for a given set of hyperparameters and the
            value of the objective function evaluated on the cross-validation set
            is returned.

            [inputs] (hyperparams_red, **kwargs)
                hyperparams_red: vector of the reduced hyperparameters
                  (see mner.solvers.manager.py)

            [returns] f_hidden
                f_hidden: numpy array of shape (1, 1) whose single
                  element is the value of the objective function
                  evaluated on the cross-validation set for the given
                  set of hyperparameters.

        """
        # initialization
        self.hyper_manager.build_state_from_red_vector(hyperparams_red.ravel(), **kwargs)
        self.hyper_manager.update_model(**kwargs)
        x0 = np.copy(self.init_x(self, **kwargs))
        hyperparams = self.hyper_manager.build_vector_from_state(**kwargs)

        if self.verbosity >= 2:
            print hyperparams_red
                    
        # solving the subproblem
        self.xtmp, self.ftmp = self.solver.solve(x0, **kwargs)
        self.fnew = self.cv_model.cost(self.xtmp.astype(self.float_dtype))

        # checking if new solution is the best found so far
        if self.fnew < self.fcv:
            self.fcv = np.copy(self.fnew)
            self.ftrain = np.copy(self.ftmp)
            self.x = np.copy(self.xtmp.astype(self.float_dtype))

        if self.verbosity >= 1:
            print "fnew = " + str(self.fnew) + ", fcv (best) = " + str(self.fcv)        

        if self.verbosity >= 3:
            print "weights:"
            print self.xtmp
    
        # update storage arrays
        if not self.forget:
            self.update_storage(self.xtmp.astype(self.float_dtype), hyperparams, self.fnew, **kwargs)

        # return cost of hidden objective function
        return self.fnew.reshape((1, 1)).astype(self.float_dtype)

    
    def solve(self, x0=None, **kwargs):
        """ Find a feasible set of hyperparameters that minimize the objective
            function evaluated on the cross-validation set and return the
            corresponding weight vector.

            [inputs] (x0=None, **kwargs)
                x0: (optional) numpy array of the weight
                  initialization (default is set to the class member
                  self.x0)

            [returns] (x, ftrain)
                x: numpy array of the weights that most closely
                  minimize the objective function.
                ftrain: scalar value of the objective function
                  evaluated at x.

        """
        # import Gaussian processes package
        import GPy
        import GPyOpt

        # initialize solution vector and costs
        self.x = np.copy(x0.astype(self.float_dtype))
        
        if not self.resume:
            # if the optimization is not being resumed
            self.ftrain = None
            self.fcv = np.inf

            # initialize regularization parameters
            if not self.forget and self.storage_initialized and self.fcv_storage.size > 0:
                hyperparams_red_init = np.zeros((self.fcv_storage.size, self.hyper_manager.red_index[-1]))
                for i in range(self.fcv_storage.size):
                    hyperparams_red_init[i, :] = self.hyper_manager.build_red_vector_from_vector(self.hyperparam_storage[i, :], **kwargs)
                fcv_init = np.copy(self.fcv_storage.reshape((self.fcv_storage.size, 1)))
            else:
                hyperparams_red_init = self.init_points(N=self.N_init, **kwargs)
                fcv_init = None
            hyperparams = self.hyper_manager.build_vector_from_red_vector(hyperparams_red_init[0, :], **kwargs)
            
            # initialize storage arrays, if applicable
            if not self.forget and not self.storage_initialized:
                self.init_storage(hyperparams, **kwargs)

            # initialize Bayesian optimization solver and populate the distribution
            if self.verbosity >= 0:
                print "Initializing Bayesian optimization."

            self.bayes_opt = GPyOpt.methods.BayesianOptimization(f=self.hidden_cost, domain=self.hyper_manager.domain_f, constrains=self.hyper_manager.cons_f, X=hyperparams_red_init, Y=fcv_init, acquisition_type=self.acquisition_type, exact_feval=self.exact_feval, normalize_Y=self.normalize_Y, acquisition_jitter=self.acquisition_jitter, model_type=self.model_type, acquisition_weight=self.exploration_weight, user_def_dist=self.hyper_manager.sampler.samp_func, model_update_interval=self.model_update_interval)

            # assume that if solve is called again, it is to resume the optimization
            self.resume = True

        # initialize kernel
        if self.kernel is None:
            self.bayes_opt.model.kernel = GPy.kern.Matern52(self.hyper_manager.red_index[-1], ARD=True)

        # final configurations before commencing optimization
        if self.kernel_variance_fixed is not None and self.kernel_variance_fixed is not False:
            self.bayes_opt.model.kernel.variance.constrain_fixed(self.kernel_variance_fixed)
        if self.kernel_lengthscale_fixed is not None and self.kernel_lengthscale_fixed is not False:
            self.bayes_opt.model.kernel.lengthscale.constrain_fixed(self.kernel_lengthscale_fixed)

        if self.verbosity >= 2:
            print "initial process parameters:"
            print self.bayes_opt.model.model

        if self.verbosity >= 0:
            print "Starting Bayesian optimization..."

        # optimization loop
        for its in range(1, self.maxits+1):
            self.bayes_opt.run_optimization(max_iter=1, eps=self.epsilon, verbosity=True)

            if self.verbosity >= 1:
                print "Bayes search iteration " + str(its)
            
            if self.verbosity >= 2:
                print "process parameters:"
                print self.bayes_opt.model.model
                self.bayes_opt._print_convergence()

            if self.verbosity >= 2:
                print ""

            if not self.bayes_opt.initial_iter and self.bayes_opt._distance_last_evaluations() < self.epsilon:
                self.converged = True
                break

        if self.verbosity >= 0:
            if self.converged:
                print "Bayesian optimization converged to desired precision."
            else:
                print "Maximum iterations exceeded in Bayesian optimization."

        return (self.x.astype(self.float_dtype), self.ftrain.astype(self.float_dtype))

    
    


class NNGlobalSearch(BaseSearch):
    """ NNGlobalSearch (class)

        Finds a globally optimal approximation to the weights subject to
        nuclear-norm regularization. The algorithm does so by
        adjusting a single (without loss of generality) nuclear-norm
        regularization parameter until the minimal amount of
        regularization necessary to reach the globally optimal
        regularization domain is achieved.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the globally optimal approximation solver.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class Optimizer
                  from module mner.optimizer.py) from which the solver
                  is instantiated.
                  - parameters are inherited from BaseSearch
                  - (nn_global_search_)rtype:
                    (default=self.train_model.rtype) list of strings
                    of regularization penalty types from
                    mner.model.py.
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - parameters are inherited from BaseSearch
                  - (nn_global_search_)rtype: same as above.
                  - (nn_global_search_)solver: (default=None)
                    subproblem solver for optimizing the objective
                    function
                  - (nn_global_search_)maxiter: (default=100) positive
                    integer defining the maximum number of iterations
                    taken by the algorithm.
                  - (nn_global_search_)ctol: (default=1.0e-4) positive
                    scalar defining the precision to which the
                    nuclear-norm regularization parameter must satisfy
                    the necessary condition for global optimality.
                  - (nn_global_search_)update_ratio: (default=0.99)
                    scalar in the range (0.0, 1.0) that determines the
                    speed of convergence. Numbers closer to the upper
                    bound converge more quickly.
                  - (nn_global_search_)verbosity: (default=1) amount of
                    information printed to screen during the
                    optimization. Options are integers between -1 and
                    3.

        """
        # initialize base class
        super(NNGlobalSearch, self).__init__(parent, **kwargs)

        # get and compile the global constraint function
        self.K = self.train_model.calc_dLdJ(x=None, **kwargs)
        self.K = theano.function(inputs=[self.train_model.x_dev], outputs=self.K)

        # set-up regularization manager
        self.rtype = kwargs.get('nn_global_search_rtype', kwargs.get('rtype', parent.get('rtype', self.train_model.rtype)))

        # get and initialize subproblem solver
        self.solver = kwargs.get('nn_global_search_solver', kwargs.get('solver', None))
        if self.solver is not None:
            self.init_solver(**kwargs)

        # get maximum number of iterations
        self.maxiter = kwargs.get('nn_global_search_maxiter', kwargs.get('maxiter', 100))
        
        # get constraint tolerance
        self.ctol = kwargs.get('nn_global_search_ctol', kwargs.get('ctol', 1.0e-4)) 

        # get update ratio
        self.update_ratio = kwargs.get('nn_global_search_update_ratio', kwargs.get('update_ratio', 0.99))

        # verbosity level
        self.verbosity = kwargs.get('nn_global_search_verbosity', kwargs.get('verbosity', 1))

        self.complete = False


    def solve(self, x0=None, **kwargs):
        """ Solve for a globally optimal approximation of the weights.

            [inputs] (x0=None, **kwargs)
                x0: (optional) numpy array of the weight
                  initialization (default is set to the class member
                  self.x0)

            [returns] (x, ftrain)
                x: numpy array of the weights that most closely
                  minimize the objective function.
                ftrain: scalar value of the objective function
                  evaluated at x.

        """
        if "nuclear-norm" not in self.rtype:
            self.state = None
            return self.solver.solve(x0, **kwargs)
        else:
            self.state = self.float_dtype(0.0)

        # bounds are defined to prevent the algorithm from infinitely jumping around when convergence is imprecise
        bounds = np.array([0.0, 1.0])
        bounds[1] = np.inf

        # in case of early stopping, makes sure solution kept is minimum feasible
        self.min_feas = np.inf

        # initialize best solution
        self.xbest = np.copy(x0)
        self.fbest = np.inf

        # initialize error signal
        self.signal = 0

        if self.verbosity >= 0:
            print "Starting globally optimal approximation algorithm..."

        # begin optimization
        for its in range(1, self.maxiter+1):
            if self.verbosity >= 1:
                print "Global opt approx iteration " + str(its)
            if self.state is not None:
                # assign nuclear-norm regularization parameter
                self.train_model.assign_reg_params("nuclear-norm", np.array([self.state]), **kwargs)

                if self.verbosity >= 2:
                    print "nuclear-norm reg. param. = " + str(self.state)

                # solve the subproblem
                self.x, self.ftrain = self.solver.solve(x0, **kwargs)

                # check the constraint
                mat = self.K(self.x)
                w = np.linalg.eigvalsh(mat)
                wmax = np.max(np.abs(w))

                if self.verbosity >= 1:
                    print "max. eval. = " + str(wmax)
                #print "state = " + str(self.state) + ", wmax = " + str(wmax)

                # adjust the regularization, if necessary
                if self.state < wmax:
                    # regularization is too small, increase
                    if self.verbosity >= 2:
                        print "nuclear-norm penalty too small, increase reg. param."

                    tmp = self.state + self.update_ratio*(wmax - self.state)
                    if tmp > bounds[1]:
                        # if eigenvalue exceeds upper bound, update using upper bound
                        self.state += self.update_ratio*(bounds[1] - self.state)
                    else:
                        # update the state using largest variance eigenvalue
                        self.state = np.copy(tmp)
                    x0 = np.copy(self.x)
                elif self.state > wmax + self.ctol:
                    # regularization is too big, decrease
                    if self.verbosity >= 2:
                        print "nuclear-norm penalty too large, decrease reg. param."

                    if self.state < self.min_feas:
                        # solution is best found so far
                        self.xbest = np.copy(self.x)
                        self.fbest = np.copy(self.ftrain)
                        self.min_feas = np.copy(self.state)
                    bounds[1] = np.copy(self.state)
                    self.state -= self.update_ratio*(self.state - wmax)
                    if self.state < bounds[0]:
                        # shouldn't happen unless convergence is imprecise
                        self.signal = -1
                        break

                    if self.verbosity >= 1:
                        print "fnew = " + str(self.ftrain) + ", ftrain (best) = " + str(self.fbest)
                else:
                    # feasible solution found
                    self.xbest = np.copy(self.x)
                    self.fbest = np.copy(self.ftrain)
                    self.min_feas = np.copy(self.state)
                    self.complete = True
                    break

            self.x = np.copy(self.xbest)
            self.ftrain = np.copy(self.fbest)

        if self.verbosity >= 0:
            if self.complete:
                print "Feasible globally optimal approximation found."
            else:
                print "Maximum number of iterations exceeded in globally optimal approximation algorithm."

        return (self.x.astype(self.float_dtype), self.ftrain.astype(self.float_dtype))


    



# global optimization heuristics
class MultiInitSearch(BaseSolver):
    """ MultiInitSearch (class)

        Search for a globally optimal solution by repeating the
        optimization from multiple initializations of the weights.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the global optimization heuristic solver.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class Optimizer
                  from module mner.optimizer.py) from which the solver
                  is instantiated.
                  - parameters are inherited from BaseSolver
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - parameters are inherited from BaseSolver
                  - (multi_init_search_)max_repeats: (default=10)
                    positive integer indicating the maximum number of
                    initializations to try.
                  - (multi_init_search_)max_fails:
                    (default=self.max_repeats) positive integer for
                    the maximum number of consecutive failures of the
                    algorithm to find a better solution. If the number
                    of failures exceeds this number, the optimization
                    is considered complete.
                  - (multi_init_search_)custom_sampling_function:
                    (default=MultiInitSearch.monte_carlo) the sampling
                    function used to generate initializations
                  - (multi_init_search_)x_origin:
                    (default=MultiInitSearch.relative_origin) set the
                    origin of the Monte Carlo search, if applicable
                  - (multi_init_search_)x_length:
                    (default=np.array([0.1])) set the step size of the
                    Monte Carlo relative to the origin, if applicable.
                  - (multi_init_search_)x_dist:
                    (default=MultiInitSearch.uniform_dist) probability
                    distribution from which initializations are drawn
                    in the Monte Carlo search about the origin
                    (x_origin) with distribution width given by the
                    step size (x_length).
                  - (multi_init_search_)solver: (default=None)
                    subproblem solver for optimizing the objective
                    function
                  - (multi_init_search_)verbosity: (default=1) amount of
                    information printed to screen during the
                    optimization. Options are integers between -1 and
                    3.

        """
        # initialize base class
        super(MultiInitSearch, self).__init__(parent, **kwargs)
        
        # get optimization parameters
        self.max_repeats = kwargs.get('multi_init_search_max_repeats', kwargs.get('max_repeats', 10))
        self.max_fails = kwargs.get('multi_init_search_max_fails', kwargs.get('max_fails', self.max_repeats))

        # origin of Monte Carlo initialization
        self.x_origin = kwargs.get('multi_init_search_x_origin', kwargs.get('x_origin', self.relative_origin))
        
        # weight step length for Monte Carlo initialization
        self.x_length = kwargs.get('multi_init_search_x_length', kwargs.get('x_length', np.array([0.1])))

        # Monte Carlo step distribution
        self.x_dist = kwargs.get('multi_init_search_x_dist', kwargs.get('x_dist', self.uniform_dist))

        # custom sampling function
        self.custom_sampling_function = kwargs.get('multi_init_search_custom_sampling_function', kwargs.get('custom_sampling_function', self.monte_carlo))

        # get subproblem solver and initialize
        self.solver = kwargs.get('multi_init_search_solver', kwargs.get('solver', None))
        if self.solver is not None:
            self.init_solver(**kwargs)

        # verbosity level
        self.verbosity = kwargs.get('multi_init_search_verbosity', kwargs.get('verbosity', 1))

        self.converged = False
            

    def uniform_dist(self, n, **kwargs):
        """ Draw a random weight vector from a unform distribution on the
            range [-1.0, 1.0].

            [inputs] (n, **kwargs)
                n: length of the weight vector.

            [returns] dx 
                dx: 1D numpy array of length n of uniformly
                  distributed random numbers.

        """
        return 0.5 * (2.0*np.random.rand(n,)-1.0)
        
        
    def relative_origin(self, xtmp, **kwargs):
        """ Set the current weight vector  as the origin.

            [inputs] (xtmp, **kwargs)
                xtmp: the current weight vector.

            [returns] xtmp
                xtmp: same as above.

        """
        return xtmp


    def monte_carlo(self, xtmp, **kwargs):
        """ Choose a random initialization about an origin from a probability
            distribution.

            [inputs] (xtmp, **kwargs)
                xtmp: the current weight vector.

            [returns] x0
                x0: new initialization of the weight vector.

        """
        return self.x_origin(xtmp=xtmp, **kwargs) + self.x_length * self.x_dist(xtmp.size, **kwargs)
    

    def solve(self, x0=None, **kwargs):
        """ Find a possible feasible global minimizer of the problem using
            multiple random initializations.

            [inputs] (x0=None, **kwargs)
                x0: (optional) numpy array of the weight
                  initialization (default is set to the class member
                  self.x0)

            [returns] (x, ftrain)
                x: numpy array of the weights that most closely
                  minimize the objective function.
                ftrain: scalar value of the objective function
                  evaluated at x.

        """
        self.fails = 0
        self.x = x0
        self.ftrain = self.train_model.cost(x0)

        if self.verbosity >= 0:
            print "Starting global optimization heuristic from multiple initializations..."

        for i in range(1, self.max_repeats+1):
            if self.verbosity >= 1:
                print "Global opt trial " + str(i)
            
            xtmp, ftmp = self.solver.solve(x0, **kwargs)
            
            if ftmp < self.ftrain:
                self.ftrain = np.copy(ftmp)
                self.x = np.copy(xtmp)
                self.fails = 0
            else:
                self.fails += 1

            if self.verbosity >= 1:
                print "# of consecutive failures = " + str(self.fails)

            if self.fails >= self.max_fails:
                self.converged = True
                break
                
            x0 = self.custom_sampling_function(xtmp=xtmp, **kwargs)

        if self.verbosity >= 0:
            if self.converged:
                print "possible global minimum found; maximum # of consecutive failures exceeded."
            else:
                print "maximum number of iterations exceeded."

        return (self.x.astype(self.float_dtype), self.ftrain.astype(self.float_dtype))


    

