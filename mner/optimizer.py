import numpy as np
import theano
import theano.tensor as T
from . import model

""" optimizer.py (module)

    The optimizer module contains the Optimizer class that is used
    to configure low-rank MNE models, initialize the solver, and 
    minimize the (possibly constrained and regularized) objective
    function.

"""

class Optimizer(object):
    """ Optimizer (class)

        The Optimizer class is the interface used for constructing
        and optimizing low-rank MNE models. This class is built
        flexibly to allow for easy customization.

    """

    def __init__(self, resp, feat, rank, cetype=None, citype=None, rtype=None, solver=None, datasets=None, **kwargs):
        """Initialize Optimizer class instantiation.

            [inputs] (resp, feat, rank, cetype=None, citype=None,
              rtype=None, solver=None, datasets=None, **kwargs)
                resp: numpy array of the output labels with shape
                  (nsamp,) where nsamp is the number of data
                  samples. Each element of resp must be in the range
                  [0, 1].
                feat: numpy array of the input features with shape
                  (nsamp, ndim) where ndim is the number of features.
                rank: positive integer that sets the number of columns
                  of the matrices U and V(that both have shape (ndim,
                  rank)).
                cetype: (optional) list of strings that tell the class
                  which equality constraints, if any, are being
                  used. Can set to None if no equality constraints are
                  used.  Available equality constraints:
                  - "UV-linear-insert": these sets each U[:,k] =
                    csigns[k]*V[:,k] for all k in range(rank) and
                    directly imposes this constraint by substitution.
                    Note that csigns is a numpy array of binary
                    integers in {-1, 1} that sets the sign
                    relationship between each component of U and V.
                    csigns may be set using **kwargs.  - "UV-linear":
                    these are the same constraints as UV-linear-insert
                    but instead of direct substitution the constraints
                    are imposed using the method Lagrange
                    multipliers. csigns must also be set through
                    **kwargs.  - "UV-quadratic": these constraints are
                    the equality constraints defined by the upper
                    triangle of np.dot(U, U.T) == np.dot(V, V.T).  -
                    "UV-bilinear": these constraints are the equality
                    constraints defined by the upper triangle (with
                    diagonal excluded) of np.dot(U, V.T) == np.dot(V,
                    U.T).
                citype: (optional) list of strings that tell the class
                  which equality constraints, if any, are being
                  used. Can set to None if no equality constraints are
                  used.  No inequality constraints are defined at this
                  time.
                rtype: (optional) list of strings that tell the class
                  which regularization penalties, if any, should be
                  added to the objective function. Can set to None if
                  no penalty functions are applied.  Available penalty
                  functions:
                  - "nuclear-norm": the nuclear-norm regularizes over
                    the Frobenius-norms of U and V and promotes
                    sparsity of the eigenvalue spectrum of J =
                    np.dot(U, V.T).
                solver: (optional) must be set and initialized (using
                  class function init_solver) before beginning the
                  optimization. It is optional to set here, however.
                datasets: (optional) is a dict with keys "trainset",
                  "cvset", "testset" with values corresponding to
                  Boolean indices for those samples that belong to
                  each the training set, cross-validation set, and
                  test set, respectively. If datasets is set to None,
                  it is assumed that all samples belong to the
                  training set and the other subsets are
                  empty. Missing fields are assumed to be empty as
                  well.

        """

        # initialize class members to standard arguments
        self.rank = rank
        self.cetype = cetype
        self.citype = citype
        self.rtype = rtype
        self.solver = solver
        self.datasets = datasets

        # get data sizes
        self.nsamp, self.ndim = self.get_data_sizes(feat)
        self.ntrain, self.ncv, self.ntest = self.get_data_subset_sample_sizes(self.nsamp, self.datasets)

        # initialize class members to keyword arguments
        self.fscale = self.get_model_scaling(kwargs.get("fscale", None))
        self.float_dtype = kwargs.get("float_dtype", np.float64)
        self.precompile = kwargs.get("precompile", True)

        # declare theano variables
        self.x_dev = T.vector("x_dev")
        self.lda_dev = T.vector("lambda_dev")

        # set-up the model(s)
        self.train_model, self.cv_model, self.test_model = self.config_models(resp, feat, **kwargs)

        # build expressions for model(s)
        self.build_expressions(self.train_model, grad=kwargs.get("compute_grad", True), hess=kwargs.get("compute_hess", False), **kwargs)
        self.build_expressions(self.cv_model, grad=False, hess=False, **kwargs)
        self.build_expressions(self.test_model, grad=False, hess=False, **kwargs)

        # compile the expressions
        if kwargs.get("precompile", True):
            self.compile_expressions(self.train_model, grad=kwargs.get("compute_grad", True), hess=kwargs.get("compute_hess", False), **kwargs)
            self.compile_expressions(self.cv_model, grad=False, hess=False, **kwargs)
            self.compile_expressions(self.test_model, grad=False, hess=False, **kwargs)

        # initilize solver
        if solver is not None:
            self.init_solver(**kwargs)

        self.initialized = True
            

    def get_data_sizes(self, feat):
        """ Get the number of samples and features.

            [inputs] (feat)
                feat: numpy array with shape (nsamp, ndim).

            [returns] (nsamp, ndim)
                nsamp: integer count of the number of samples in the
                  data set.
                ndim: integer count of the number of features in the
                  data set.

        """
        return feat.shape

    
    def get_data_subset_sample_sizes(self, nsamp, datasets):
        """ Get the number of samples in each of the data subsets.

            [inputs] (nsamp, datasets)
                nsamp: integer count of the number of samples in the
                  data set.
                datasets: see definition from class function __init__.

            [returns] (ntrain, ncv, ntest)
                ntrain: integer count of the number of samples in the
                  training set.
                ncv: integer count of the number of samples in the
                  cross-validation set.
                ntest: integer count of the number of samples in the
                  test set.

        """
        ntrain, ncv, ntest = 0, 0, 0
        if "trainset" in datasets:
            ntrain = np.sum(datasets["trainset"])
            if "cvset" in datasets:
                ncv = np.sum(datasets["cvset"])
            if "testset" in datasets:
                ntest = np.sum(datasets["testset"])
        else:
            ntrain = nsamp

        return (ntrain, ncv, ntest)

    
    def get_model_scaling(self, fscale, **kwargs):
        """ Determine the scaling of the negative log-likelihood objective
            function (from mner.model.py).

            [inputs] (fscale, **kwargs)
                fscale: dict with keys "trainset", "cvset", and
                  "testset" with values that give the rescaling of the
                  objective function for the training set,
                  cross-validation set, and test set, respectively. If
                  a value is set to <=0 then the objective function is
                  scaled by the number of samples in each data
                  subset. If a value is set to None, then the
                  objective function is unscaled.

            [returns] fscale
                fscale: see inputs.

        """
        if fscale is None:
            fscale = dict()
            if "trainset" in self.datasets:
                fscale["trainset"] = 1.0
                if "cvset" in self.datasets:
                    fscale["cvest"] = 1.0
                if "testset" in self.datasets:
                    fscale["testset"] = 1.0
            else:
                fscale["trainset"] = 1.0
        else:
            if not isinstance(fscale, dict):
                if isinstance(fscale, list) or isinstance(fscale, tuple):
                    tmp = fscale.copy()
                    fscale = dict()
                    idx = 0
                    if "trainset" in self.datasets:
                        fscale["trainset"] = fscale[idx]
                        idx += 1
                        if "cvset" in self.datasets:
                            fscale["cvset"] = fscale[idx]
                            idx += 1
                        if "testset" in self.datasets:
                            fscale["testset"] = fscale[idx]
                            idx += 1
                else:
                    fscale = {"trainset": fscale, "cvset": fscale, "testset": fscale}
        
        # if the scaling is set to a negative number, set scaling to 1.0/samples
        if "trainset" in fscale and fscale["trainset"] <= 0.0:
            fscale["trainset"] = 1.0/self.ntrain
        else:
            fscale["trainset"] = 1.0
        if "cvset" in fscale and fscale["cvset"] <= 0.0:
            fscale["cvset"] = 1.0/self.ncv
        else:
            fscale["cvset"] = 1.0
        if "testset" in fscale and fscale["testset"] <= 0.0:
            fscale["testset"] = 1.0/self.ntest
        else:
            fscale["testset"] = 1.0
            
        return fscale
            

    def config_models(self, resp, feat, fscale=None, **kwargs):
        """ Configure the low-rank MNE model(s) by instantiating the class
            MNEr from mner.model.py.

            [inputs] (resp, feat, fscale=None, **kwargs)
                resp: see the class function __init__
                feat: see the class function __init__
                fscale: (optional) see the class function
                  get_model_scaling

            [returns] (train_model, cv_model, test_model)
                train_model: training set instantiation of class MNEr
                  with any regularization and constraints imposed.
                cv_model: cross-validation set instantiation of class
                  MNEr (unregularized and no constraints)
                test_model: test set instantiation of class MNEr
                  (unregularized and no constraints)

        """
        self.use_vars = kwargs.get("use_vars", {'avar': True, 'hvar': True, 'UVvar': True})
        self.use_consts = kwargs.get("use_consts", {'aconst': False, 'hconst': False, 'UVconst': False, 'Jconst': False})
        
        train_model, cv_model, test_model = None, None, None
        if self.datasets is None:
            # model trained on entire dataset
            train_model = model.MNEr(resp, feat, self.rank, cetype=self.cetype, citype=self.citype, rtype=self.rtype, fscale=self.fscale["trainset"], use_vars=self.use_vars, use_consts=self.use_consts, float_dtype=self.float_dtype, x_dev=self.x_dev, **kwargs)
        else:
            # model trained on subset of dataset
            if "trainset" in self.datasets:
                train_model = model.MNEr(resp[self.datasets["trainset"]], feat[self.datasets["trainset"],:], self.rank, cetype=self.cetype, citype=self.citype, rtype=self.rtype, fscale=self.fscale["trainset"], use_vars=self.use_vars, use_consts=self.use_consts, float_dtype=self.float_dtype, x_dev=self.x_dev, **kwargs)
            if "cvset" in self.datasets:
                cv_model = model.MNEr(resp[self.datasets["cvset"]], feat[self.datasets["cvset"],:], self.rank, cetype=self.cetype, citype=self.citype, fscale=self.fscale["cvset"], use_vars=self.use_vars, use_consts=self.use_consts, float_dtype=self.float_dtype, x_dev=self.x_dev, **kwargs)
            if "testset" in self.datasets:
                test_model = model.MNEr(resp[self.datasets["testset"]], feat[self.datasets["testset"],:], self.rank, cetype=self.cetype, citype=self.citype, fscale=self.fscale["testset"], use_vars=self.use_vars, use_consts=self.use_consts, float_dtype=self.float_dtype, x_dev=self.x_dev, **kwargs)

        return (train_model, cv_model, test_model)

    
    def build_expressions(self, model, grad=True, hess=False, **kwargs):
        """Build Theano expressions for the objective, constraints, gradient,
            Jacobians, and Hessian, if applicable, for a given model.

            [inputs] (model, grad=True, hess=False, **kwargs)
                model: instantiation of class MNEr from mner.model.py
                grad: (optional) Boolean; if True, builds the
                  gradient.
                hess: (optional) Boolean; if True, builds the Hessian.

        """
        if model is not None:
            # build cost expression (gradient and hessian, if applicable)
            # note that regularization is included in the cost expression
            model.cost_expr(self.x_dev)
            if grad:
                model.cost_grad_expr(self.x_dev)
            if hess:
                model.cost_hess_expr(self.x_dev)

            # build equality constraints expressions (gradient and hessian, if applicable)
            if model.num_lagrange_cetypes:
                model.ceq_expr(self.x_dev)
                if grad:
                    model.ceq_jaco_expr(self.x_dev)
                if hess:
                    model.ceq_hess_expr(self.x_dev, self.lda_dev)

            # build inequality constraints expressions (gradient and hessian, if applicable)
            if model.num_lagrange_citypes:
                model.cineq_expr(self.x_dev)
                if grad:
                    model.cineq_jaco_expr(self.x_dev)
                if hess:
                    model.cineq_hess_expr(self.x_dev)


    def compile_expressions(self, model, grad=True, hess=False, **kwargs):
        """Compile Theano expressions into device functions for a given
            model.

            [inputs] (model, grad=True, hess=False, **kwargs)
                model: instantiation of class MNEr from mner.model.py
                grad: (optional) Boolean; if True, compiles the
                  gradient
                hess: (optional) Boolean; if True, compiles the
                  Hessian

        """
        if model is not None:
            # compile cost function (gradient and hessian, if applicable)
            # note that this cost function includes regularization
            model.compile_cost(self.x_dev)
            if grad:
                model.compile_cost_grad(self.x_dev)
            if hess:
                model.compile_cost_hess(self.x_dev)

            # compile equality constraints (gradient and hessian, if applicable)
            if model.cetype is not None and len(model.cetype):
                model.compile_ceq(self.x_dev)
                if grad:
                    model.compile_ceq_jaco(self.x_dev)
                if hess:
                    model.compile_ceq_hess(self.x_dev, self.lda_dev)
            
            # compile inequality constraints (gradient and hessian, if applicable)
            if model.citype is not None and len(model.citype):
                model.compile_cineq(self.x_dev)
                if grad:
                    model.compile_cineq_jaco(self.x_dev)
                if hess:
                    model.compile_cineq_hess(self.x_dev, self.lda_dev)


    def init_solver(self, **kwargs):
        """ Initialize the solver object.

            [inputs] (**kwargs)

        """
        if not hasattr(self.solver, 'initialized') or self.solver.initialized == False:
            self.solver = self.solver(self.__dict__, **kwargs)
            

    def compute_set(self, set_name, **kwargs):
        """ Compute the objective function at the current weights.

            [inputs] (set_name, **kwargs)
                set_name: string with the name of the data set;
                  i.e. "train", "cv", "test.

            [returns] fval
                fval: value of the objective function evaluated on the
                  set_name data set.

        """
        #if set_name in self.datasets:
        if set_name.endswith("set"):
            set_name = set_name[:-len("set")]
        fval = eval("self." + set_name + "_model.cost(self.x.astype(self.float_dtype))")

        return fval


    def optimize(self, x0=None, **kwargs):
        """Optimize the low-rank MNE model.

            [inputs] (x0=None, **kwargs)
                x0: (optional) initial weight vector. If set to None,
                  then the class function init_vec from the class
                  member train_model is used.

            [returns] (x, ftrain)
                x: final weight vector after the optimization
                  completes
                ftrain: value of the (regularized and constrained)
                  objective function at x from class member
                  train_model.

        """
        # if initial weights are not provided, generate some
        if x0 is None:
            x0 = self.train_model.init_vec()

        self.x, self.ftrain = self.solver.solve(x0.astype(self.float_dtype), **kwargs)
        return (self.x, self.ftrain)
