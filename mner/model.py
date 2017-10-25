import numpy as np
import theano.tensor as T
import theano
from scipy.linalg import svd

""" model.py (module)

    This module contains the basic model classes, their objective
    functions, gradients, constraints, and Hessians. For now, the only
    model is the 'MNEr' model for the low-rank MNE method. Full-rank
    second-order MNE and first- order MNE may be included at some
    point in the future.

"""

# low-rank MNE model class
class MNEr:
    """ MNEr (class)

        This class contains the basic necessities for optimizing a
        low-rank MNE model including the objective function, gradient,
        Hessian, constraints, and regularization.

    """


    # class initialization
    def __init__(self, resp, feat, rank, cetype=None, citype=None, rtype=None, fscale=1.0, use_vars={'avar': True, 'hvar': True, 'UVvar': True}, use_consts={'aconst': False, 'hconst': False, 'UVconst': False, 'Jconst': False}, **kwargs):
        """Initialize MNEr class instantiation.

            Initialize the class instantiation and some basic set-up
            of important parameters. It is best to include as much of
            the input parameters as possible at this stage if they
            differ from the defaults.

            [inputs] (resp, feat, rank, cetype=None, citype=None,
              rtype=None, fscale=1.0, use_vars={'avar': True, 'hvar':
              True, 'UVvar': True}, use_consts={'aconst': False,
              'hconst': False, 'UVconst': False, 'Jconst': False},
              **kwargs)
                resp: numpy array of the output labels with shape
                  (nsamp,) where nsamp is the number of data
                  samples. Each element of resp must be in the range
                  [0, 1].
                feat: numpy array of the input features with shape
                  (nsamp, ndim) where ndim is the number of features.
                rank: positive integer that sets the number of columns
                  of the matrices U and V (that both have shape (ndim,
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
                    csigns may be set using **kwargs.
                  - "UV-linear": these are the same constraints as
                    UV-linear-insert but instead of direct
                    substitution the constraints are imposed using the
                    method Lagrange multipliers. csigns must also be
                    set through **kwargs.
                  - "UV-quadratic": these constraints are the equality
                    constraints defined by the upper triangle of
                    np.dot(U, U.T) == np.dot(V, V.T).
                  - "UV-bilinear": these constraints are the equality
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
                fscale: (optional) a float scalar that rescales the
                  negative log-likelihood objective function (but not
                  the regularization penalty functions).
                use_vars: (optional) a dictionary that signals whether
                  these variables appear in the argument of the
                  logistic function. The gradient, Hessian, and any
                  constraints are taken with respect to these
                  variables.
                use_consts: (optional) a dictionary that signals
                  whether these constants (device variables) should
                  appear in the objective. Note that the gradient,
                  Hessian, and constraints are NOT taken with respect
                  to these variables.

                kwargs: keyword arguments are used to set
                  model-specific hyperparameters and floating point
                  data type.
                  - float_dtype: (optional) the floating- point data
                    type of data and mathematical operations.
                  - csigns: (optional, default=None) numpy array of
                    values drawn from {-1, 1} that set the sign
                    relationship betweens columns in U and V for the
                    constraints "UV-linear" and "UV-linear-insert"
                    (required when using these
                    constraints). I.e. U[:,k] = csigns[k]*V[:,k].

        """

        # getting floating point data type
        self.float_dtype = kwargs.get("float_dtype", np.float64)

        # get number samples
        self.nsamp = resp.size
        # get number of features
        self.ndim = feat.shape[1]
        # assert that the feature array has the same number of samples as the labels
        assert self.nsamp == feat.shape[0]

        # load the labels onto the device
        self.resp = theano.shared(resp.reshape((self.nsamp,)).astype(self.float_dtype), name="resp")
        # load the features onto the device
        self.feat = theano.shared(feat.astype(self.float_dtype), name="feat")
        
        # get the rank of the low-rank MNE model
        self.rank = rank

        # get the list of equality constraints to be applied, if any
        if cetype is None:
            self.cetype = []
        elif isinstance(cetype, str):
            self.cetype = [cetype]
        else:
            self.cetype = cetype
        # initialize the counter of which equality constraints will be applied via the method of Lagrange multipliers
        self.num_lagrange_cetypes = 0

        # get the list of inequality constraints to be applied, if any
        if citype is None:
            self.citype = []
        elif isinstance(citype, str):
            self.citype = [citype]
        else:
            self.citype = citype
        # initialize the counter of which inequality constraints will be applied via the method of Lagrange multipliers
        self.num_lagrange_citypes = 0

        # get the list of regularization penalties to be applied through addition of a penalty function to the objective function, if any
        if rtype is None:
            self.rtype = [] 
        elif isinstance(rtype, str):
            self.rtype = [rtype]
        else:
            self.rtype = rtype

        # get the scale of the negative log-likelihood objective function
        self.fscale = fscale

        # get the dictionary signalling which variables to optimize with respect to and set defaults if they do not exist in the dictionary
        self.use_vars = use_vars
        if 'avar' not in self.use_vars:
            self.use_vars['avar'] = True
        if 'hvar' not in self.use_vars:
            self.use_vars['hvar'] = True
        if 'UVvar' not in self.use_vars:
            self.use_vars['UVvar'] = True

        # get the dictionary signalling whether some additional constants should be included in the model (these are not optimized) and fill any missing keys with the default values
        self.use_consts = use_consts
        if 'aconst' not in self.use_consts:
            self.use_consts['aconst'] = False
        if 'hconst' not in self.use_consts:
            self.use_consts['hconst'] = False
        if 'UVconst' not in self.use_consts:
            self.use_consts['UVconst'] = False
        if 'Jconst' not in self.use_consts:
            self.use_consts['Jconst'] = False

        # get machine precision for the given floating-point precision
        self.eps = np.finfo(self.float_dtype).eps

        # count the number of weights that need to be optimized
        self.nvar = 0
        if self.use_vars['avar']:
            self.nvar += 1
        if self.use_vars['hvar']:
            self.nvar += self.ndim
        if self.use_vars['UVvar']:
            if "UV-linear-insert-relaxed" in self.cetype:
                self.nvar += self.rank*(self.ndim+1)
            elif "UV-linear-insert" in self.cetype:
                self.nvar += self.rank*self.ndim
            elif "UV-linear-relaxed" in self.cetype:
                self.nvar += self.rank*(2*self.ndim+1)
            else:
                self.nvar += 2*self.rank*self.ndim

        # initialize constants on the device
        if self.use_consts['aconst']:
            self.aconst = theano.shared(np.zeros((1,)).astype(self.float_dtype), name="aconst")
        if self.use_consts['hconst']:
            self.hconst = theano.shared(np.zeros((self.ndim,)).astype(self.float_dtype), name="hconst")
        if self.use_consts['UVconst']:
            self.Uconst = theano.shared(np.zeros((self.ndim, 1)).astype(self.float_dtype), name="Uconst")
            self.Vconst = theano.shared(np.zeros((self.ndim, 1)).astype(self.float_dtype), name="Vconst")
        if self.use_consts['Jconst']:
            self.Jconst = theano.shared(np.zeros((self.ndim, self.ndim)).astype(self.float_dtype), name="Jconst")

        # get signs for the linear equality constraints and initialize on the device
        csigns = kwargs.get("csigns", None)
        if isinstance(csigns, list) or isinstance(csigns, tuple):
            csigns = np.array(csigns).astype(self.float_dtype)
        if csigns is not None:
            self.init_csigns(csigns)

        # get x_dev, if provided
        self.x_dev = kwargs.get('x_dev', None)

        # initialize all Theano expressions that may be used in the optimization to None
        self.f = None
        self.df = None
        self.d2f = None
        self.ce = None
        self.dce = None
        self.d2ce = None
        self.ci = None
        self.dci = None
        self.d2ci = None

        # initialize all compiled functions that may be used in the optimization to None
        self.cost = None
        self.grad = None
        self.hess = None
        self.ceq = None
        self.ceq_jaco = None
        self.ceq_hess = None
        self.cineq = None
        self.cineq_jaco = None
        self.cineq_hess = None
        

    # miscellaneous functions
    def UV_to_Q(self, U, V, **kwargs):
        """ Construct Q from U and V where Q via concatenation.

             [inputs] (U, V, **kwargs)
                U: numpy array with shape (ndim, rank) or Theano
                  tensor/expression.
                V: numpy array with shape (ndim, rank) or Theano
                  tensor/expression.

             [returns] Q
                Q: numpy array with shape (2*ndim, rank) or Theano
                  tensor/expression.

        """
        if isinstance(U, np.ndarray) and isinstance(V, np.ndarray):
            return np.concatenate([U, V], axis=0)
        else:
            return T.concatenate([U, V], axis=0)


    def Q_to_UV(self, Q, **kwargs):
        """ Construct U and V from Q.

            [inputs] (Q, **kwargs)
                Q: numpy array with shape (2*ndim, rank) unless
                  "UV-linear-insert" equality constraint is used in
                  which case the shape is (ndim, rank) or a Theano
                  tensor/expression.

            [returns] (U, V)
                U: numpy array with shape (ndim, rank) or Theano
                  tensor/expression.
                V: numpy array with shape (ndim, rank) or Theano
                  tensor/expression.

        """
        if Q is None:
            return (None, None)
        else:
            U = Q[:self.ndim, :]
            if "UV-linear-insert-relaxed" in self.cetype:
                if isinstance(U, np.ndarray):
                    V = np.copy(U)
                else:
                    V = U
            elif "UV-linear-insert" in self.cetype:
                if isinstance(U, np.ndarray):
                    V = U * np.tile(self.csigns.reshape((1, self.rank)), (self.ndim, 1))
                else:
                    V = U * T.tile(self.csigns.reshape((1, self.rank)), (self.ndim, 1))
            else:
                V = Q[self.ndim:, :]

            return (U, V)


    def decompose_J(self, J, **kwargs):
        """ Decompose J using the singular value decomposition.

            [inputs] (J, **kwargs)
                J: numpy array with shape (ndim, ndim) representing J
                  = np.dot(U, V.T).

            [returns] (L, S, R)
                L: numpy array of left eigenvectors of J with shape
                  (ndim, ndim) where the columns are the eigenvectors.
                S: numpy array of a diagonal matrix of singular values
                  with shape (ndim, ndim).
                R: numpy array of right eigenvectors of J with shape
                  (ndim, ndim) where the rows are the eigenvectors.

        """
        L, S, R = svd(0.5*(J+J.T))
        return (L, S, R)

    
    def J_to_UV(self, J, **kwargs):
        """ Construct U and V through singular value decomposition of J.

            [inputs] (J, **kwargs)
                J: numpy array with shape (ndim, ndim) representing J
                  = np.dot(U, V.T).

            [returns] (U, V)
                U: numpy array with shape (ndim, rank)
                V: numpy array with shape (ndim, rank)

        """
        L, S, R = self.decompose_J(J)
        U = np.dot(L, np.diag(np.sqrt(S)))
        V = np.dot(R.T, np.diag(np.sqrt(S)))
        return (U, V)


    def J_to_Q(self, J, **kwargs):
        """ Construct Q through singular value decomposition of J.

            [inputs] (J, **kwargs) 
                J: numpy array with shape (ndim, ndim) representing J
                  = np.dot(U, V.T).

            [returns] Q
                Q: numpy array with shape (2*ndim, rank)

        """
        U, V = self.J_to_UV(J)
        Q = np.concatenate([U, V], axis=0)
        return Q


    def UV_to_J(self, U, V, **kwargs):
        """ Construct J from the outer product of matrices U and V.

            [inputs] (U, V, **kwargs)
                U: numpy array with shape (ndim, rank)
                V: numpy array with shape (ndim, rank)

            [returns] J
                J: numpy array with shape (ndim, ndim) representing J
                  = np.dot(U, V.T) (symmetrized).

        """
        J = np.dot(U, V.T)
        return 0.5*(J+J.T)


    def Q_to_J(self, Q, **kwargs):
        """ Construct J from Q.

            [inputs] (Q, **kwargs)
                Q: numpy array with shape (2*ndim, rank)

            [returns]
                J: numpy array with shape (ndim, ndim) representing J
                  = np.dot(U, V.T) (symmetrized).

        """
        ndim = Q.shape[0]/2
        return self.UV_to_J(Q[:ndim, :], Q[ndim:, :], **kwargs)


    def vec_to_weights(self, x, **kwargs):
        """ Convert a weight vector to a scalar, vector, matrix, and a possible
            empty additional vector.

            [inputs] (x, **kwargs)
                x: numpy array with shape (nvar,) or a Theano
                  tensor/expression

            [returns] (a, h, Q, w)
                a: numpy array with shape (1,) or Theano expression.
                h: numpy array with shape (ndim,) or Theano expression
                Q: numpy array with shape (2*ndim, rank) unless
                  "UV-linear-insert" equality constraint is used in
                  which case the shape is (ndim, rank) or a Theano
                  tensor/expression
                w: numpy array with variable shape or Theano
                  expression corresponding to additional
                  weights. Returns None if not used.

        """
        offset = 0
        if self.use_vars['avar']:
            a = x[offset]
            offset += 1
        else:
            a = None
        if self.use_vars['hvar']:
            h = x[offset:self.ndim+offset]
            offset += self.ndim
        else:
            h = None
        if self.use_vars['UVvar']:
            if "UV-linear-insert-relaxed" in self.cetype:
                Q = x[offset:offset+self.rank*self.ndim].reshape((self.rank, self.ndim)).T
                offset += self.rank*self.ndim
                w = x[offset:offset+self.rank]
                offset += self.rank
            elif "UV-linear-insert" in self.cetype:
                Q = x[offset:offset+self.rank*self.ndim].reshape((self.rank, self.ndim)).T
                offset += self.rank*self.ndim
                w = None
            elif "UV-linear-relaxed" in self.cetype:
                Q = x[offset:offset+2*self.rank*self.ndim].reshape((self.rank, 2*self.ndim)).T
                offset += 2*self.rank*self.ndim
                w = x[offset:offset+self.rank]
                offset += self.rank
            else:
                Q = x[offset:offset+2*self.rank*self.ndim].reshape((self.rank, 2*self.ndim)).T
                offset += 2*self.rank*self.ndim
                w = None
        else:
            Q = None
            w = None

        return (a, h, Q, w)


    def weights_to_vec(self, a, h, Q, **kwargs):
        """ Convert a scalar, vector, and matrix to a weight vector.

            [inputs] (a, h, Q, **kwargs)
                a: numpy array with shape (1,) or Theano expression.
                h: numpy array with shape (ndim,) or Theano expression
                Q: numpy array with shape (2*ndim, rank) unless
                  "UV-linear-insert" equality constraint is used in
                  which case the shape is (ndim, rank) or a Theano
                  tensor/expression

            [returns] x
                x: numpy array with shape (nvar,) or a Theano
                  tensor/expression

        """
        w = kwargs.get('w', None)
        x = np.zeros((self.nvar,), dtype=self.float_dtype)
        offset = 0
        if self.use_vars['avar']:
            x[offset] = a
            offset += 1
        if self.use_vars['hvar']:
            x[offset:self.ndim+offset] = h
            offset += self.ndim
        if self.use_vars['UVvar']:
            x[offset:offset+Q.size] = Q.T.reshape((Q.size,))
            if "UV-linear-insert-relaxed" in self.cetype:
                offset += self.rank * self.ndim
                x[offset:offset+self.rank] = w
                offset += self.rank
            elif "UV-linear-insert" in self.cetype:
                offset += self.rank * self.ndim
            elif "UV-linear-relaxed" in self.cetype:
                offset += 2*self.rank * self.ndim
                x[offset:offset+self.rank] = w
                offset += self.rank
            else:
                offset += 2*self.rank * self.ndim

        return x    


    def assign_aconst(self, a, **kwargs):
        """ Set value of aconst on device.

            [inputs] (a, **kwargs)
                a: numpy array with shape (1,)

        """
        self.aconst.set_value(a.astype(self.float_dtype))


    def update_aconst(self, da, **kwargs):
        """ Update value of aconst on device.

            [inputs] (da, **kwargs)
                da: numpy array with shape (1,)

        """
        self.assign_aconst(self.aconst.get_value() + da.astype(self.float_dtype))
    

    def assign_hconst(self, h, **kwargs):
        """ Set value of hconst on device.

            [inputs] (h, **kwargs)
                h: numpy array with shape (ndim,)

        """
        self.hconst.set_value(h.astype(self.float_dtype))


    def update_hconst(self, dh, **kwargs):
        """ Update value of hconst on device.

            [inputs] (dh, **kwargs)
                dh: numpy array with shape (ndim,)

        """
        self.assign_hconst(self.hconst.get_value() + dh.astype(self.float_dtype))


    def assign_Jconst(self, U=None, V=None, J=None, **kwargs):
        """ Set value of Jconst on device.

            [inputs] (U=None, V=None, J=None, **kwargs)
                U: (optional) numpy array with shape (ndim, rank)
                V: (optional) numpy array with shape (ndim, rank)
                J: (optional) numpy array with shape (ndim, ndim)
                  representing J = np.dot(U, V.T)

        """
        if J is None:
            assert U is not None
            assert V is not None

            self.Jconst.set_value(np.dot(U, V.T).astype(self.float_dtype))
        else:
            assert J is not None

            self.Jconst.set_value(J.astype(self.float_dtype))


    def update_Jconst(self, dU=None, dV=None, dJ=None, **kwargs):
        """ Update value of Jconst on device.

            [inputs] (dU=None, dV=None, dJ=None, **kwargs)
                dU: (optional) numpy array with shape (ndim, rank)
                dV: (optional) numpy array with shape (ndim, rank)
                dJ: (optional) numpy array with shape (ndim, ndim)
                  representing dJ = np.dot(dU, dV.T)

        """
        if dJ is None:
            assert dU is not None
            assert dV is not None

            if dU.size == dU.shape[0]:
                dU = dU.reshape((self.ndim, 1))
            if dV.size == dV.shape[0]:
                dV = dV.reshape((self.ndim, 1))
            
            self.Jconst.set_value(self.Jconst.get_value() + np.dot(dU, dV.T).astype(self.float_dtype))
        else:
            assert dJ is not None

            self.Jconst.set_value(self.Jconst.get_value() + dJ.astype(self.float_dtype))


    def assign_Uconst(self, U=None, J=None, **kwargs):
        """ Set value of Uconst on device.

            [inputs] (U=None, J=None, **kwargs)
                U: (optional) numpy array with shape (ndim, rank)
                J: (optional) numpy array with shape (ndim, ndim)
                  representing J = np.dot(U, V.T)

        """
        if J is None:
            assert U is not None
            self.Uconst.set_value(U)
        else:
            assert J is not None
            U, V = self.J_to_UV(J)
            self.Uconst.set_value(U)


    def assign_Vconst(self, V=None, J=None, **kwargs):
        """ Set value of Vconst on device.

            [inputs] (V=None, J=None, **kwargs)
                V: (optional) numpy array with shape (ndim, rank)
                J: (optional) numpy array with shape (ndim, ndim)
                  representing J = np.dot(U, V.T)

        """
        if J is None:
            assert V is not None
            self.Vconst.set_value(V)
        else:
            assert J is not None
            U, V = self.J_to_UV(J)
            self.Vconst.set_value(V)


    def assign_UVconst(self, U=None, V=None, J=None, **kwargs):
        """ Set values of Uconst and Vconst on device.

            [inputs] (U=None, V=None, J=None, **kwargs)
                U: (optional) numpy array with shape (ndim, rank)
                V: (optional) numpy array with shape (ndim, rank)
                J: (optional) numpy array with shape (ndim, ndim)
                  representing J = np.dot(U, V.T)

        """
        if J is None:
            assert U is not None
            assert V is not None
            self.Uconst.set_value(U)
            self.Vconst.set_value(V)
        else:
            assert J is not None
            U, V = self.J_to_UV(J)
            self.Uconst.set_value(U)
            self.Vconst.set_value(V)


    def update_Uconst(self, dU=None, dJ=None, **kwargs):
        """ Update value of Uconst on device.

            [inputs] (dU=None, dJ=None, **kwargs)
                dU: (optional) numpy array with shape (ndim, rank)
                dJ: (optional) numpy array with shape (ndim, ndim)
                  representing dJ = np.dot(dU, dV.T)

        """
        if dJ is None:
            assert dU is not None
            self.Uconst.set_value(self.Uconst.get_value() + dU)
        else:
            assert dJ is not None
            dU, dV = self.J_to_UV(dJ)
            self.Uconst.set_value(self.Uconst.get_value() + dU)


    def update_Vconst(self, dV=None, dJ=None, **kwargs):
        """ Update value of Vconst on device.

            [inputs] (dV=None, dJ=None, **kwargs)
                dV: (optional) numpy array with shape (ndim, rank)
                dJ: (optional) numpy array with shape (ndim, ndim)
                  representing dJ = np.dot(dU, dV.T)

        """
        if dJ is None:
            assert dV is not None
            self.Vconst.set_value(self.Vconst.get_value() + dV)
        else:
            assert dJ is not None
            dU, dV = self.J_to_UV(dJ)
            self.Vconst.set_value(self.Vconst.get_value() + dV)


    def update_UVconst(self, dU=None, dV=None, dJ=None, **kwargs):
        """ Update values of Uconst and Vconst on device.

            [inputs] (dU=None, dV=None, dJ=None, **kwargs)
                dU: (optional) numpy array with shape (ndim, rank)
                dV: (optional) numpy array with shape (ndim, rank)
                dJ: (optional) numpy array with shape (ndim, ndim)
                  representing dJ = np.dot(dU, dV.T)

        """
        if dJ is None:
            assert dU is not None
            assert dV is not None
            self.Uconst.set_value(self.Uconst.get_value() + dU)
            self.Vconst.set_value(self.Vconst.get_value() + dV)
        else:
            assert dJ is not None
            dU, dV = self.J_to_UV(dJ)
            self.Uconst.set_value(self.Uconst.get_value() + dU)
            self.Vconst.set_value(self.Vconst.get_value() + dV)


    def assign_reg_params(self, rtype, vals, **kwargs):
        """ Set value of regularization parameter(s) of a regularization
            penalty type.

            [inputs] (rtype, vals, **kwargs)
                rtype: string naming one regularization type
                vals: numpy array of regularization parameters to
                  assign to the rtype penalty function

        """
        size = kwargs.get('size', 1)
        idx = self.rtype.index(rtype)
        if size > 1:
            self.reg_params[idx].set_value(np.tile(vals.reshape((vals.size,1)), (size, 1)).ravel().astype(self.float_dtype))
        else:
            self.reg_params[idx].set_value(vals.astype(self.float_dtype))
        
        
    def init_csigns(self, csigns, **kwargs):
        """ Initialize csigns for the linear equality constraints.  Do NOT use
            this change csigns because the new values will be
            ignored. Use class function assign_csigns or update_csigns
            instead.

            [inputs]
                csigns: numpy array of binary values {-1, 1} with
                  shape (rank,)

        """
        if isinstance(csigns, list) or isinstance(csigns, tuple):
            csigns = np.array(csigns)
        self.csigns = theano.shared(csigns.astype(self.float_dtype), name="csigns")

        
    def assign_csigns(self, csigns, **kwargs):
        """ Set csigns for the linear equality constraints.

            [inputs]
                csigns: numpy array of binary values {-1, 1} with
                  shape (rank,)

        """
        if isinstance(csigns, list) or isinstance(csigns, tuple):
            csigns = np.array(csigns)
        self.csigns.set_value(csigns.astype(self.float_dtype))

        
    def update_csigns(self, U=None, V=None, J=None, **kwargs):
        """ Update csigns for the linear equality constraints by analyzing the
            signs of either U and V or J.

            [inputs] (U=None, V=None, J=None, **kwargs)
                U: (optional) numpy array with shape (ndim, rank)
                V: (optional) numpy array with shape (ndim, rank)
                J: (optional) numpy array with shape (ndim, ndim)
                  representing J = np.dot(U, V.T)

        """
        if J is None:
            assert U is not None
            assert V is not None
            csigns = []
            for i in range(U.shape[1]):
                if np.dot(U[:, i], V[:, i]) >= 0:
                    csigns.append(1)
                else:
                    csigns.append(-1)
            csigns = np.array(csigns, dtype=self.float_dtype)
            self.assign_csigns(csigns)
        else:
            assert J is not None
            U, V = self.J_to_UV(J)
            self.update_csigns(U=U, V=V, J=None)

            
    def unroll_triu(self, idx, v, M, k, **kwargs):
        """ Unroll the upper triangle of a Theano tensor.

            [inputs] (idx, v, M, k, **kwargs)
                idx: integer row index
                v: Theano tensor/expression storing the unrolled upper
                  triangle of M of shape (ndim*(ndim+1)/2,)
                M: Theano tensor/expression of a matrix with shape
                  (ndim, ndim)
                k: integer index correponding to the offest of the
                  diagonal.

            [returns] v
                v: Theano tensor/expression storing the unrolled upper
                  triangle of M of shape (ndim*(ndim+1)/2,)

        """
        idxA = idx*self.ndim-(idx-1)*idx/2-k*idx
        idxB = (idx+1)*self.ndim-idx*(idx+1)/2-k*(idx+1)
        v = T.set_subtensor(v[idxA:idxB], M[idx, idx+k:])
        return v


    # default weight initialization
    def init_weights(self, **kwargs):
        """ Initialize a, h, Q, and x (and w) to internal variables.

            [inputs] (**kwargs)

            [returns] (a, h, Q, w)
                a: numpy array with shape (1,)
                h: numpy array with shape (ndim,)
                Q: numpy array with shape (2*ndim, rank) unless
                  "UV-linear-insert" equality constraint is used in
                  which case the shape is (ndim, rank)
                w: numpy array with variable shape or Theano
                  expression corresponding to additional
                  weights. Returns None if not used.

        """
        if self.use_vars['avar']:
            self.a = (0.001 * np.random.randn(1,)).astype(self.float_dtype)
        else:
            self.a = None
        if self.use_vars['hvar']:
            self.h = (0.001 * np.random.randn(self.ndim,)).astype(self.float_dtype)
        else:
            self.h = None
        if self.use_vars['UVvar']:
            if "UV-linear-insert-relaxed" in self.cetype:
                self.Q = (0.01 * np.random.randn(self.ndim, self.rank)).astype(self.float_dtype)
                self.w = (np.random.randint(0, 2, size=(self.rank,))*2.0 - 1.0).astype(self.float_dtype)
            elif "UV-linear-insert" in self.cetype:
                self.Q = (0.01 * np.random.randn(self.ndim, self.rank)).astype(self.float_dtype)
                self.w = None
            elif "UV-linear-relaxed" in self.cetype:
                self.Q = (0.01 * np.random.randn(2*self.ndim, self.rank)).astype(self.float_dtype)
                self.w = (np.random.randint(0, 2, size=(self.rank,))*2.0 - 1.0).astype(self.float_dtype)
            else:
                self.Q = (0.01 * np.random.randn(2*self.ndim, self.rank)).astype(self.float_dtype)
                self.w = None
        else:
            self.Q = None
            self.w = None

        self.x = self.weights_to_vec(self.a, self.h, self.Q, w=self.w)
        return (self.a, self.h, self.Q, self.w)


    def init_weights_feas(self, **kwargs):
        """ Initialize feasible a, h, Q, and x (and w) to internal variables.

            [inputs] (**kwargs)

            [returns] (a, h, Q, w)
                a: numpy array with shape (1,)
                h: numpy array with shape (ndim,)
                Q: numpy array with shape (2*ndim, rank) unless
                  "UV-linear-insert" equality constraint is used in
                  which case the shape is (ndim, rank)
                w: numpy array with variable shape or Theano
                  expression corresponding to additional
                  weights. Returns None if not used.

        """
        if self.use_vars['avar']:
            self.a = (0.001 * np.random.randn(1,)).astype(self.float_dtype)
        else:
            self.a = None
        if self.use_vars['hvar']:
            self.h = (0.001 * np.random.randn(self.ndim,)).astype(self.float_dtype)
        else:
            self.h = None
        if self.use_vars['UVvar']:
            if "UV-linear-insert-relaxed" in self.cetype:
                self.Q = (0.01 * np.random.randn(self.ndim, self.rank)).astype(self.float_dtype)
                self.w = (np.random.randint(0, 2, size=(self.rank,))*2.0 - 1.0).astype(self.float_dtype)
            if "UV-linear-insert" in self.cetype:
                self.Q = (0.01 * np.random.randn(self.ndim, self.rank)).astype(self.float_dtype)
                self.w = None
            else:
                self.Q = (0.01 * np.random.randn(2*self.ndim, self.rank)).astype(self.float_dtype)

                if "UV-linear-relaxed" in self.cetype:
                    self.w = (np.random.randint(0, 2, size=(self.rank,))*2.0 - 1.0).astype(self.float_dtype)
                    U, V = self.Q_to_UV(self.Q)
                    V = U * np.tile(self.w.reshape((1, self.rank)), (U.shape[0], 1))
                    self.Q = self.UV_to_Q(U, V)
                elif "UV-linear" in self.cetype:
                    # requires self.csigns be set
                    U, V = self.Q_to_UV(self.Q)
                    V = U * np.tile(self.csigns.get_value().reshape((1, self.csigns.get_value().size)), (U.shape[0], 1))
                    self.Q = self.UV_to_Q(U, V)
                elif "UV-bilinear" in self.cetype:
                    U, V = self.Q_to_UV(self.Q)
                    s = np.random.rand(1, self.rank)
                    s[s >= 0.5] = 1.0
                    s[s < 0.5] = -1.0
                    V = U * np.tile(s, (U.shape[0], 1))
                    self.Q = self.UV_to_Q(U, V)
                    self.w = None
                elif "UV-quadratic" in self.cetype:
                    U, V = self.Q_to_UV(self.Q)
                    s = np.random.rand(1, self.rank)
                    s[s >= 0.5] = 1.0
                    s[s < 0.5] = -1.0
                    V = U * np.tile(s, (U.shape[0], 1))
                    self.Q = self.UV_to_Q(U, V)
                    self.w = None
                elif "UV-inner-product":
                    U, V = self.Q_to_UV(self.Q)
                    s = np.random.rand(1, self.rank)
                    s[s >= 0.5] = 1.0
                    s[s < 0.5] = -1.0
                    V = U * np.tile(s, (U.shape[0], 1))
                    self.Q = self.UV_to_Q(U, V)
                    self.w = None
        else:
            self.Q = None
            self.w = None

        self.x = self.weights_to_vec(self.a, self.h, self.Q, w=self.w)
        return (self.a, self.h, self.Q, self.w)


    def init_vec(self, **kwargs):
        """ Intialize a, h, Q, and x (and w) to internal variables.

            [inputs] (**kwargs)

            [returns] x
                x: numpy array with shape (nvar,)

        """
        self.init_weights()
        return self.x


    def init_vec_feas(self, **kwargs):
        """ Initialize feasible a, h, Q, and x (and w) to internal
            variables. Returns x.

            [inputs] (**kwargs)

            [returns] x
                x: numpy array with shape (nvar,)

        """
        self.init_weights_feas()
        return self.x


    # Theano symbolic expressions
    def arg_expr(self, x=None, **kwargs):
        """ Construct an expression for the argument of the probability
            (logistic) function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] arg
                arg: Theano expression.

        """
        if x is None:
            x = self.x_dev

        a, h, Q, w = self.vec_to_weights(x)
        if self.use_vars['UVvar']:
            U, V = self.Q_to_UV(Q)
        if "UV-linear-insert-relaxed" in self.cetype:
            V *= T.tile(w.reshape((1, self.rank)), (self.ndim, 1))
            
        if self.use_vars['avar']:
            atot = a
        else:
            atot = 0.0

        if self.use_vars['hvar']:
            htot = h.reshape((self.ndim, 1))
        else:
            htot = 0.0

        if self.use_vars['UVvar']:
            Jtot = T.dot(U, V.T)
        else:
            Jtot = 0.0

        if self.use_consts['aconst']:
            atot += self.aconst

        if self.use_consts['hconst']:
            htot += self.hconst.reshape((self.ndim, 1))

        if self.use_consts['UVconst'] and self.use_consts['Jconst']:
            Jtot += T.dot(self.Uconst, self.Vconst.T) + self.Jconst
        elif self.use_consts['UVconst']:
            Jtot += T.dot(self.Uconst, self.Vconst.T)
        elif self.use_consts['Jconst']:
            Jtot += self.Jconst

        arg = T.zeros((self.resp.size, 1))
        if self.use_vars['avar'] or self.use_consts['aconst']:
            arg += atot
        if self.use_vars['hvar'] or self.use_consts['hconst']:
            arg += T.dot(self.feat, htot)
        if self.use_vars['UVvar'] or self.use_consts['UVconst'] or self.use_consts['Jconst']:
            arg += T.sum(self.feat * T.dot(self.feat, Jtot), axis=1).reshape((self.resp.size, 1))

        arg = arg.reshape((self.resp.size,))

        self.arg = arg
        return arg


    def prob_expr(self, x=None, **kwargs):
        """ Construct an expression for the probability (logistic) function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] P
                P: Theano expression.

        """
        if x is None:
            x = self.x_dev

        arg = self.arg_expr(x)
        P = 1.0/(1.0 + T.exp(-arg))

        self.P =  P
        return P


    def calc_dLdJ(self, x=None, **kwargs):
        """ Construct an expression for the matrix derivative of the negative
            log-likelihood function with respect to "J". This can be
            used in making a globally optimal low-rank approximation
            to J.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] K
                K: Theano expression.

        """
        if x is None:
            x = self.x_dev

        P = self.prob_expr(x)

        # returns 'K'
        return T.dot(self.feat.T * (P - self.resp), self.feat) * self.fscale


    def reg_expr(self, x=None, **kwargs):
        """ Construct an expression for the regularization penalty functions.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] l
                l: Theano expression.

        """
        if x is None:
            x = self.x_dev

        a, h, Q, w = self.vec_to_weights(x)
        if self.use_vars['UVvar']:
            U, V = self.Q_to_UV(Q)
        if "UV-linear-insert-relaxed" in self.cetype:
            V *= T.tile(w.reshape((1, self.rank)), (self.ndim, 1))

        l = 0.0
        self.reg_params = [[]]*len(self.rtype)
        if "nuclear-norm" in self.rtype:
            idx = self.rtype.index("nuclear-norm")
            self.reg_params[idx] = theano.shared(np.zeros((self.rank,)).astype(self.float_dtype), name="reg_params_nuclear-norm")
            if "UV-linear-insert-relaxed" in self.cetype:
                l += 0.5 * T.sum( (U ** 2 + V ** 2) * T.tile(self.reg_params[idx].reshape((1, self.rank)), (self.ndim, 1)) )
            elif "UV-linear-insert" in self.cetype:
                l += 0.5 * T.sum( (U ** 2 + V ** 2) * T.tile(self.reg_params[idx].reshape((1, self.rank)), (self.ndim, 1)) )
            else:
                l += 0.5 * T.sum( (Q ** 2) * T.tile(self.reg_params[idx].reshape((1, self.rank)), (2*self.ndim, 1)) )

        if "l2-norm" in self.rtype:
            idx = self.rtype.index("l2-norm")
            self.reg_params[idx] = theano.shared(np.zeros((1,)).astype(self.float_dtype), name="reg_param_l2-norm")
            l += 0.5 * self.reg_params[idx] * T.sum(h ** 2)

        self.l = l
        return l


    def reg_grad_expr(self, x=None, **kwargs):
        """ Construct an expression for the gradient of the regularization
            penalty functions.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] dl
                dl: Theano expression.

        """
        if x is None:
            x = self.x_dev

        a, h, Q, w = self.vec_to_weights(x)
        if self.use_vars['UVvar']:
            U, V = self.Q_to_UV(Q)
        if "UV-linear-insert-relaxed" in self.cetype:
            V *= T.tile(w.reshape((1, self.rank)), (self.ndim, 1))

        dl = T.zeros((self.nvar,))
        if "nuclear-norm" in self.rtype:
            idx = self.rtype.index("nuclear-norm")
            offset = 0
            if self.use_vars['avar']:
                offset += 1
            if self.use_vars['hvar']:
                offset += self.ndim
            if "UV-linear-insert-relaxed" in self.cetype:
                dldQ = (U * T.tile((T.ones((1, self.rank)) + w.reshape((1, self.rank)))*self.reg_params[idx].reshape((1, self.rank)), (self.ndim, 1))).T.reshape((U.size, 1))
                dldw = 0.5 * T.sum( U ** 2 * T.tile(self.reg_params[idx].reshape((1, self.rank)), (self.ndim, 1)), axis=0).T.reshape((w.size, 1))
            elif "UV-linear-insert" in self.cetype:
                dldQ = (2.0 * U * T.tile(self.reg_params[idx].reshape((1, self.rank)), (self.ndim, 1))).T.reshape((U.size, 1))
            else: 
                dldQ = (Q * T.tile(self.reg_params[idx].reshape((1, self.rank)), (2*self.ndim, 1))).T.reshape((Q.size, 1))
            if offset > 0:
                if "UV-linear-insert-relaxed" in self.cetype:
                    dl += T.concatenate([T.zeros((offset, 1)), dldQ, dldw], axis=0).reshape((self.nvar,))
                elif "UV-linear-relaxed" in self.cetype:
                    dl += T.concatenate([T.zeros((offset, 1)), dldQ, T.zeros((w.size, 1))], axis=0).reshape((self.nvar,))
                else:
                    dl += T.concatenate([T.zeros((offset, 1)), dldQ], axis=0).reshape((self.nvar,))
            else:
                if "UV-linear-insert-relaxed" in self.cetype:
                    dl += T.concatenate([dldQ, dldw], axis=0).reshape((self.nvar,))
                elif "UV-linear-relaxed" in self.cetype:
                    dl += T.concatenate([dldQ, T.zeros((w.size, 1))], axis=0).reshape((self.nvar,))
                else:
                    dl += dldQ.reshape((self.nvar,))

        if "l2-norm" in self.rtype:
            idx = self.rtype.index("l2-norm")
            offset = 0
            if self.use_vars['avar']:
                offset += 1
            if self.use_vars['hvar']:
                dl = T.inc_subtensor(dl[offset:offset+self.ndim], self.reg_params[idx][0] * h)

        self.dl = dl
        return dl


    def reg_hess_expr(self, x=None, **kwargs):
        """ Construct an expression for the Hessian of the regularization
            penalty functions.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] d2l
                d2l: Theano expression.

        """
        if x is None:
            x = self.x_dev

        a, h, Q, w = self.vec_to_weights(x)
        if self.use_vars['UVvar']:
            U, V = self.Q_to_UV(Q)

        d2l = T.zeros((self.nvar, self.nvar))
        if "nuclear-norm" in self.rtype:
            idx = self.rtype.index("nuclear-norm")
            offset = 0
            if self.use_vars['avar']:
                offset += 1
            if self.use_vars['hvar']:
                offset += self.ndim
            if "UV-linear-insert" in self.cetype or "UV-linear-insert-relaxed" in self.cetype:
                d2ldQ2 = 2.0*T.nlinalg.diag(T.tile(self.reg_params[idx].reshape((self.rank, 1)), (1, self.ndim)).reshape((self.rank*self.ndim,)))
            else:
                d2ldQ2 = T.nlinalg.diag(T.tile(self.reg_params[idx].reshape((self.rank, 1)), (1, 2*self.ndim)).reshape((2*self.rank*self.ndim,)))
            d2l = T.inc_subtensor(d2l[offset:offset+d2ldQ2.shape[0], offset:offset+d2ldQ2.shape[1]], d2ldQ2)

        if "l2-norm" in self.rtype:
            idx = self.rtype.index("l2-norm")
            offset = 0
            if self.use_vars['avar']:
                offset += 1
            if self.use_vars['hvar']:
                d2l = T.inc_subtensor(d2l[offset:offset+self.ndim, offset:offset+self.ndim], T.nlinalg.diag(T.tile(self.reg_params[idx].reshape((1, 1)), (self.ndim, 1)).ravel()))

        self.d2l = d2l
        return d2l


    def cost_expr(self, x=None, **kwargs):
        """ Construct an expression for the (regularized) negative
            log-likelihood objective (cost) function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] f
                f: Theano expression.

        """
        if x is None:
            x = self.x_dev

        arg = self.arg_expr(x)

        Z0 = 1.0 + T.exp(arg)
        Z1 = 1.0 + T.exp(-arg)

        f = T.sum(self.resp * T.log(Z1 + self.eps) + (1.0 - self.resp) * T.log(Z0 + self.eps)) * self.fscale

        if self.rtype is not None:
            l = self.reg_expr(x)
            f += l

        self.f = f
        return f
        

    def cost_grad_expr(self, x=None, **kwargs):
        """ Construct an expression for the gradient of the (regularized)
            negative log-likelihood objective function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] df
                df: Theano expression.

        """
        if x is None:
            x = self.x_dev

        a, h, Q, w = self.vec_to_weights(x)
        if self.use_vars['UVvar']:
            U, V = self.Q_to_UV(Q)
        if "UV-linear-insert-relaxed" in self.cetype:
            V *= T.tile(w.reshape((1, self.rank)), (self.ndim, 1))

        P = self.prob_expr(x)

        df = T.zeros((self.nvar,))
        offset = 0
        if self.use_vars['avar']:
            df = T.set_subtensor(df[offset], T.sum(P - self.resp) * self.fscale)
            offset += 1
        if self.use_vars['hvar']:
            df = T.set_subtensor(df[offset:self.ndim+offset], T.dot(self.feat.T, P-self.resp) * self.fscale)
            offset += self.ndim        

        if self.use_vars['UVvar']:
            C = self.calc_dLdJ(x)
            if "UV-linear-insert-relaxed" in self.cetype:
                df = T.set_subtensor(df[offset:offset+self.rank*self.ndim], 2.0 * T.dot(C, V).T.reshape((Q.size,))) # fscale already included in C
                offset += self.rank*self.ndim
                df = T.set_subtensor(df[offset:offset+self.rank], T.dot((T.dot(self.feat, U) ** 2).T, P-self.resp) * self.fscale)
                offset += self.rank
            elif "UV-linear-insert" in self.cetype:
                df = T.set_subtensor(df[offset:offset+self.rank*self.ndim], 2.0 * T.dot(C, V).T.reshape((Q.size,))) # fscale already included in C
                offset += self.rank*self.ndim
            else:
                X1 = T.concatenate([T.zeros((self.ndim, self.ndim)), C], axis=1)
                X2 = T.concatenate([C, T.zeros((self.ndim, self.ndim))], axis=1)
                X = T.concatenate([X1, X2], axis=0)

                df = T.set_subtensor(df[offset:offset+2*self.rank*self.ndim], T.dot(X, Q).T.reshape((Q.size,))) # fscale already included in C
                offset += 2*self.rank*self.ndim

        if self.rtype is not None:
            dl = self.reg_grad_expr(x)
            df += dl

        self.df = df
        return df


    def cost_hess_expr(self, x=None, **kwargs):
        """ Construct an expression for the Hessian of the (regularized)
            negative log-likelihood objective function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] d2f
                d2f: Theano expression.

        """
        if x is None:
            x = self.x_dev

        offset = 0
        if self.use_vars['avar']:
            offset += 1
        if self.use_vars['hvar']:
            offset += self.ndim

        def block_vec(idx, g, U, V):
            g = T.set_subtensor(g[:, offset+2*idx*self.ndim:offset+(1+2*idx)*self.ndim], T.tile(T.dot(self.feat, V[:, idx]).reshape((self.resp.size, 1)), (1, self.ndim)) * self.feat)
            g = T.set_subtensor(g[:, offset+(1+2*idx)*self.ndim:offset+(2+2*idx)*self.ndim], T.tile(T.dot(self.feat, U[:, idx]).reshape((self.resp.size, 1)), (1, self.ndim)) * self.feat)

            return g

        def block_vec_ins(idx, g, V):
            g = T.set_subtensor(g[:, offset+idx*self.ndim:offset+(1+idx)*self.ndim], 2.0 * T.tile(T.dot(self.feat, V[:, idx]).reshape((self.resp.size, 1)), (1, self.ndim)) * self.feat)

            return g

        def block_mat(idx, Xblk, X):
            idx1 = offset+2*idx*self.ndim
            idx2 = offset+2*(idx+1)*self.ndim
            Xblk = T.set_subtensor(Xblk[idx1:idx2, idx1:idx2], X)

            return Xblk

        def block_mat_ins(idx, Xblk, X):
            idx1 = offset+idx*self.ndim
            idx2 = offset+(1+idx)*self.ndim
            Xblk = T.set_subtensor(Xblk[idx1:idx2, idx1:idx2], 2.0 * X * self.csigns[idx])

            return Xblk

        def block_relax_ins(idx, Xblk, X, U, w):
            idx1 = offset+idx*self.ndim
            idx2 = offset+(1+idx)*self.ndim
            Xblk = T.set_subtensor(Xblk[idx1:idx2, idx1:idx2], 2.0 * X * w[idx])
            idx3 = offset+self.rank*self.ndim+idx
            Xblk = T.set_subtensor(Xblk[idx3], 2.0 * (T.dot(X, U[:,idx]) ** 2))

            return Xblk
            
        a, h, Q, w = self.vec_to_weights(x)
        if self.use_vars['UVvar']:
            U, V = self.Q_to_UV(Q)
        if "UV-linear-insert-relaxed" in self.cetype:
            V *= T.tile(w.reshape((1, self.rank)), (self.ndim, 1))

        if self.use_vars['UVvar']:
            if "UV-linear-insert" in self.cetype or "UV-linear-insert-relaxed" in self.cetype:
                g, updates = theano.scan(
                                fn=block_vec_ins,
                                outputs_info=T.zeros((self.resp.size, self.nvar)),
                                sequences=T.arange(U.shape[1]),
                                non_sequences=V,
                            )
            else:
                g, updates = theano.scan(
                                fn=block_vec,
                                outputs_info=T.zeros((self.resp.size, self.nvar)),
                                sequences=T.arange(U.shape[1]),
                                non_sequences=[U, V],
                            )

            g = g[-1]
        else:
            g = T.zeros((self.resp.size, self.nvar))

        offset = 0
        if self.use_vars['avar']:
            g = T.set_subtensor(g[:, offset], T.ones((self.resp.size,)))
            offset += 1
        if self.use_vars['hvar']:
            g = T.set_subtensor(g[:, offset:self.ndim+offset], self.feat)
            offset += self.ndim
        if "UV-linear-insert-relaxed" in self.cetype:
            offset += self.rank * self.ndim
            g = T.set_subtensor(g[:, offset:offset+self.rank], T.dot(self.feat, U) ** 2)
            offset += self.rank

        P = self.prob_expr(x)

        d2f = T.dot(g.T * (P * (1 - P)), g) * self.fscale

        if self.use_vars['UVvar']:
            C = self.calc_dLdJ(x) # fscale already included in C
            if "UV-linear-insert-relaxed" in self.cetype:
                Xfull, updates = theano.scan(
                                    fn=block_relax_ins,
                                    outputs_info=T.zeros((self.nvar, self.nvar)),
                                    sequences=T.arange(U.shape[1]),
                                    non_sequences=[C, U, w],
                                )
            elif "UV-linear-insert" in self.cetype:
                Xfull, updates = theano.scan(
                                    fn=block_mat_ins,
                                    outputs_info=T.zeros((self.nvar, self.nvar)),
                                    sequences=T.arange(U.shape[1]),
                                    non_sequences=C,
                                )
            else:
                X1 = T.concatenate([T.zeros((self.ndim, self.ndim)), C], axis=1)
                X2 = T.concatenate([C, T.zeros((self.ndim, self.ndim))], axis=1)
                X = T.concatenate([X1, X2], axis=0)

                Xfull, updates = theano.scan(
                                    fn=block_mat,
                                    outputs_info=T.zeros((self.nvar, self.nvar)),
                                    sequences=T.arange(Q.shape[1]),
                                    non_sequences=X,
                                )

            d2f += Xfull[-1]

        if self.rtype is not None:
            d2l = self.reg_hess_expr(x)
            d2f += d2l

        self.d2f = d2f
        return d2f


    def ceq_expr(self, x=None, **kwargs):
        """ Construct an expression for the equality constraint functions.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] ce
                ce: Theano expression.

        """
        if x is None:
            x = self.x_dev

        a, h, Q, w = self.vec_to_weights(x)
        if self.use_vars['UVvar']:
            U, V = self.Q_to_UV(Q)
        
        ce = T.as_tensor_variable(np.array([], dtype=self.float_dtype).reshape((0, 1)))
        if "UV-linear-relaxed" in self.cetype:
            # requires usage of w
            clr = (U - V * T.tile(w.reshape((1, self.rank)), (self.ndim, 1))).T.reshape((self.rank*self.ndim, 1))
            ce = T.concatenate([ce, clr], axis=0)
            self.num_lagrange_cetypes += 1
        elif "UV-linear" in self.cetype:
            # requires that self.csigns be set such that V[:, k] = +/-1*U[:, k] = self.csigns[k]*U[:,k] where self.csigns is a device variable
            clin = (U - V * T.tile(self.csigns.reshape((1, self.rank)), (self.ndim, 1))).T.reshape((self.rank*self.ndim, 1))
            ce = T.concatenate([ce, clin], axis=0)
            self.num_lagrange_cetypes += 1
        elif "UV-bilinear" in self.cetype:
            cbilin = T.triu(T.dot(U, V.T) - T.dot(V, U.T), k=1)

            cbilin_unr, updates = theano.scan(
                fn=self.unroll_triu,
                outputs_info=T.zeros((self.ndim*(self.ndim+1)/2-self.ndim,)),
                sequences=T.arange(self.ndim),
                non_sequences=[cbilin, 1],
            )

            cbilin_unr = cbilin_unr[-1].reshape((self.ndim*(self.ndim+1)/2-self.ndim,1))
            ce = T.concatenate([ce, cbilin_unr], axis=0)
            self.num_lagrange_cetypes += 1
        elif "UV-quadratic" in self.cetype:
            cquad = T.triu(T.dot(U, U.T) - T.dot(V, V.T))
            
            cquad_unr, updates = theano.scan(
                fn=self.unroll_triu,
                outputs_info=T.zeros((self.ndim*(self.ndim+1)/2,)),
                sequences=T.arange(self.ndim),
                non_sequences=[cquad, 0],
            )

            cquad_unr = cquad_unr[-1].reshape((self.ndim*(self.ndim+1)/2,1))
            ce = T.concatenate([ce, cquad_unr], axis=0)
            self.num_lagrange_cetypes += 1
        elif "UV-inner-product" in self.cetype:
            cinner = (T.sum(U * V, axis=0)**2 - T.sum(U ** 2, axis=0) * T.sum(V ** 2, axis=0)).reshape((self.rank, 1))
            ce = T.concatenate([ce, cinner], axis=0)
            self.num_lagrange_cetypes += 1
            
        ce = ce.reshape((ce.size,))

        self.ce = ce
        return ce


    def ceq_jaco_expr(self, x=None, **kwargs):
        """ Construct an expression for the equality constraint Jacobians.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] dce
                dce: Theano expression.

        """
        if x is None:
            x = self.x_dev

        a, h, Q, w = self.vec_to_weights(x)
        if self.use_vars['UVvar']:
            U, V = self.Q_to_UV(Q)

        dce = T.as_tensor_variable(np.array([], dtype=self.float_dtype).reshape((self.nvar, 0)))
        # automatic differentiation works reasonably well for these
        if "UV-linear-relaxed" in self.cetype:
            # requires usage of w
            clr = (U - V * T.tile(w.reshape((1, self.rank)), (self.ndim, 1))).T.reshape((self.rank*self.ndim,))
            dclr = theano.gradient.jacobian(clr, wrt=x).reshape((self.rank*self.ndim, self.nvar)).T
            dce = T.concatenate([dce, dclr], axis=1)
        elif "UV-linear" in self.cetype:
            # requires that self.csigns be set such that V[:, k] = +/-1*U[:, k] = self.csigns[k]*U[:,k] where self.csigns is a device variable
            clin = (U - V * T.tile(self.csigns.reshape((1, self.rank)), (self.ndim, 1))).T.reshape((self.rank*self.ndim,))
            dclin = theano.gradient.jacobian(clin, wrt=x).reshape((self.rank*self.ndim, self.nvar)).T
            dce = T.concatenate([dce, dclin], axis=1)
        elif "UV-bilinear" in self.cetype:
            cbilin = T.triu(T.dot(U, V.T) - T.dot(V, U.T), k=1)

            cbilin_unr, updates = theano.scan(
                fn=self.unroll_triu,
                outputs_info=T.zeros((self.ndim*(self.ndim+1)/2-self.ndim,)),
                sequences=T.arange(self.ndim),
                non_sequences=[cbilin, 1],
            )

            cbilin_unr = cbilin_unr[-1]
            dcbilin = theano.gradient.jacobian(cbilin_unr, wrt=x).reshape((self.ndim*(self.ndim+1)/2-self.ndim, self.nvar)).T
            dce = T.concatenate([dce, dcbilin], axis=1)
        elif "UV-quadratic" in self.cetype:
            cquad = T.triu(T.dot(U, U.T) - T.dot(V, V.T))

            cquad_unr, updates = theano.scan(
                fn=self.unroll_triu,
                outputs_info=T.zeros((self.ndim*(self.ndim+1)/2,)),
                sequences=T.arange(self.ndim),
                non_sequences=[cquad, 0],
            )

            cquad_unr = cquad_unr[-1]
            dcquad = theano.gradient.jacobian(cquad_unr, wrt=x).reshape((self.ndim*(self.ndim+1)/2, self.nvar)).T
            dce = T.concatenate([dce, dcquad], axis=1)
        elif "UV-inner-product" in self.cetype:
            cinner = T.sum(U * V, axis=0)**2 - T.sum(U ** 2, axis=0) * T.sum(V ** 2, axis=0)
            dcinner = theano.gradient.jacobian(cinner, wrt=x).reshape((self.rank, self.nvar)).T
            dce = T.concatenate([dce, dcinner], axis=1)

        self.dce = dce
        return dce


    def ceq_hess_expr(self, x=None, lda=None, **kwargs):
        """ Construct an expression for the equality constraint Hessians.

            [inputs] (x=None, lda=None, **kwargs)
                x: (optional) Theano tensor for the weights.
                lda: (optional) Theano tensor for all Lagrange
                  multipliers.

            [returns] d2ce
                d2ce: Theano expression.

        """
        if x is None:
            x = self.x_dev
        if lda is None:
            lda = self.lda_dev

        a, h, Q, w = self.vec_to_weights(x)
        U, V = self.Q_to_UV(Q)

        # using automatic differentiation
        d2ce = theano.gradient.hessian(cost=T.sum(self.ce * lda[:self.ce.size]), wrt=x)

        self.d2ce = d2ce
        return d2ce


    def cineq_expr(self, x=None, **kwargs):
        """ Construct an expression for the inequality constraint functions.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] ci
                ci: Theano expression.

        """
        if x is None:
            x = self.x_dev

        a, h, Q, w = self.vec_to_weights(x)
        U, V = self.Q_to_UV(Q)

        # did not need this
        ci = None

        self.ci = ci
        return ci


    def cineq_jaco_expr(self, x=None, **kwargs):
        """ Construct an expression for the inequality constraint Jacobians.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] dci
                dci: Theano expression.

        """
        if x is None:
            x = self.x_dev

        a, h, Q, w = self.vec_to_weights(x)
        U, V = self.Q_to_UV(Q)

        # did not need this
        dci = None

        self.dci = dci
        return dci


    def cineq_hess_expr(self, x=None, lda=None, **kwargs):
        """ Construct an expression for the inequality constraint Hessians.

            [inputs] (x=None, lda=None, **kwargs)
                x: (optional) Theano tensor for the weights.
                lda: (optional) Theano tensor for all Lagrange
                  multipliers.

            [returns] d2ci
                d2ci: Theano expression.

        """
        if x is None:
            x = self.x_dev
        if lda is None:
            lda = self.lda_dev

        a, h, Q, w = self.vec_to_weights(x)
        U, V = self.Q_to_UV(Q)

        # did not need this
        d2ci = None

        self.d2ci = d2ci
        return d2ci


    # compile Theano expressions
    def compile_arg(self, x=None, **kwargs):
        """ Compile the argument of the probability (logistic) function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] argument
                argument: compiled function.

        """
        if x is None:
            x = self.x_dev

        self.argument = theano.function(
                            inputs=[x],
                            outputs=self.arg,
                            on_unused_input='ignore',
                        )

        return self.argument


    def compile_prob(self, x=None, **kwargs):
        """ Compile the probability (logistic) function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] probability
                probability: compiled function.

        """
        if x is None:
            x = self.x_dev

        self.probability = theano.function(
                                inputs=[x],
                                outputs=self.P,
                                on_unused_input='ignore',
                            )

        return self.probability


    def compile_cost(self, x=None, **kwargs):
        """ Compile the (regularized) objective function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] cost
                cost: compiled function.

        """
        if x is None:
            x = self.x_dev

        self.cost = theano.function(
            inputs=[x],
            outputs=self.f,
            on_unused_input='ignore',
        )

        return self.cost

        
    def compile_cost_grad(self, x=None, **kwargs):
        """ Compile the gradient of the (regularized) objective function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] grad
                grad: compiled function.

        """
        if x is None:
            x = self.x_dev

        self.grad = theano.function(
            inputs=[x],
            outputs=self.df,
            on_unused_input='ignore',
        )

        return self.grad


    def compile_cost_hess(self, x=None, **kwargs):
        """ Compile the Hessian of the (regularized) objective function.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] hess
                hess: compiled function.

        """
        if x is None:
            x = self.x_dev

        self.hess = theano.function(
            inputs=[x],
            outputs=self.d2f,
            on_unused_input='ignore',
        )

        return self.hess


    def compile_ceq(self, x=None, **kwargs):
        """ Compile the equality constraint functions.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] ceq
                ceq: Compiled function.

        """
        if x is None:
            x = self.x_dev

        self.ceq = theano.function(
            inputs=[x],
            outputs=self.ce,
            on_unused_input='ignore',
        )

        return self.ceq


    def compile_ceq_jaco(self, x=None, **kwargs):
        """ Compile the equality constraint Jacobians.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] ceq_jaco
                ceq_jaco: Compiled function.

        """
        if x is None:
            x = self.x_dev

        self.ceq_jacobian = theano.function(
            inputs=[x],
            outputs=self.dce,
            on_unused_input='ignore',
        )

        return self.ceq_jaco


    def compile_ceq_hess(self, x=None, lda=None, **kwargs):
        """ Compile the equality constraint Hessians.

            [inputs] (x=None, lda=None, **kwargs)
                x: (optional) Theano tensor for the weights.
                lda: (optional) Theano tensor for all the Lagrange
                  multipliers.

            [returns] ceq_hess
                ceq_hess: compiled function.

        """
        if x is None:
            x = self.x_dev
        if lda is None:
            lda = self.lda_dev

        self.ceq_hess = theano.function(
            inputs=[x, lda],
            outputs=self.d2ce,
            on_unused_input='ignore',
        )

        return self.ceq_hess


    def compile_cineq(self, x=None, **kwargs):
        """ Compile the inequality constraint functions.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] cineq
                cineq: compiled function.

        """
        if x is None:
            x = self.x_dev

        self.cineq = theano.function(
            inputs=[x],
            outputs=self.ci,
            on_unused_input='ignore',
        )

        return self.cineq


    def compile_cineq_jaco(self, x=None, **kwargs):
        """ Compile the inequality constraint Jacobians.

            [inputs] (x=None, **kwargs)
                x: (optional) Theano tensor for the weights.

            [returns] cineq_jaco
                cineq_jaco: compiled function.

        """
        if x is None:
            x = self.x_dev

        self.cineq_jacobian = theano.function(
            inputs=[x],
            outputs=self.dci,
            on_unused_input='ignore',
        )

        return self.cineq_jaco


    def compile_cineq_hess(self, x=None, lda=None, **kwargs):
        """ Compile the inequality constraint Hessians.

            [inputs] (x=None, lda=None, **kwargs)
                x: (optional) Theano tensor for the weights.
                lda: (optional) Theano tensor for all the Lagrange
                  multiplers.

            [returns] cineq_hess
                cineq_hess: Compiled function.

        """
        if x is None:
            x = self.x_dev
        if lda is None:
            lda = self.lda_dev

        self.cineq_hess = theano.function(
            inputs=[x, lda],
            outputs=self.d2ci,
            on_unused_input='ignore',
        )

        return self.cineq_hess








# main function for testing
if __name__ == "__main__":

    def nice_print(M):
        for i in range(M.shape[0]):
            msg = ""
            for j in range(M.shape[1]):
                if len(str(M[i,j])) > 6:
                    smsg = str(M[i,j])[:6]
                else:
                    smsg = str(M[i,j])
                msg += smsg + "\t"
            print msg        


    print "----------------------------"
    print "UNIT TESTING"
    print "----------------------------"
    print ""

    resp = np.array([0.1, 0.5, 0.7, 0.2])
    feat = np.array([[1.0, -1.0, 0.5], [0.5, -0.25, -1.0], [-0.75, 0.75, 1.5], [-2.0, 0.25, -1.0]])

    rank = 2
    cetype = "UV-linear-insert"
    citype = None
    rtype = ["nuclear-norm", "l2-norm"]
    fscale = 1.0/resp.size
    use_vars = {'avar': True, 'hvar': True, 'UVvar': True}
    use_consts = {'aconst': True, 'hconst': True, 'UVconst': True, 'Jconst': True}
    if theano.config.floatX == "float64":
        float_dtype = np.float64
    elif theano.config.floatX == "float32":
        float_dtype = np.float32
    else:
        float_dtype = np.float64
    
    if rtype is not None:
        reg_params = [[]]*len(rtype)
    else:
        reg_params = []
    if "nuclear-norm" in rtype:
        idx = rtype.index("nuclear-norm")
        reg_params[idx] = np.random.rand(rank,)
    if "l2-norm" in rtype:
        idx = rtype.index("l2-norm")
        reg_params[idx] = np.random.rand(1,)
    #reg_params = np.arange(0.1, 0.1*(rank+0.5), 0.1)
    csigns = []
    for i in range(rank):
        if i % 2 == 0:
            csigns.append(1)
        else:
            csigns.append(-1)
    csigns = np.array(csigns)

    print "nuclear-norm regularization parameters:"
    print reg_params
    print ""

    x = T.vector('x')
    lda = T.vector('lda')

    if cetype == "UV-linear-relaxed":
        lda_const = np.random.randn(feat.shape[1]*rank).astype(float_dtype)
    elif cetype == "UV-linear":
        lda_const = np.random.randn(feat.shape[1]*rank).astype(float_dtype)
    elif cetype == "UV-bilinear":
        lda_const = np.random.randn(feat.shape[1]*(feat.shape[1]+1)/2-feat.shape[1]).astype(float_dtype)
    elif cetype == "UV-quadratic":
        lda_const = np.random.randn(feat.shape[1]*(feat.shape[1]+1)/2).astype(float_dtype)
    elif cetype == "UV-inner-product":
        lda_const = np.random.randn(rank).astype(float_dtype)
        
    print "constructing and compiling functions..."
    model = MNEr(resp, feat, rank, cetype, citype, rtype, fscale, use_vars, use_consts, float_dtype)
    model.init_csigns(csigns)
    model.cost_expr(x)
    model.compile_cost(x)

    model.cost_grad_expr(x)
    model.compile_cost_grad(x)

    model.cost_hess_expr(x)
    model.compile_cost_hess(x)
    
    if cetype is not None and "UV-linear-insert" not in model.cetype and "UV-linear-insert-relaxed" not in model.cetype and use_vars['UVvar']:
        model.ceq_expr(x)
        model.compile_ceq(x)

        model.ceq_jaco_expr(x)
        model.compile_ceq_jaco(x)

        model.ceq_hess_expr(x, lda)
        model.compile_ceq_hess(x, lda)
    print "complete."
    print ""

    #model.init_weights_feas()
    model.init_weights()
    #model.x += float_dtype(0.25)
    # showing bilinear rank-deficiency
    #a, h, Q, _ = model.vec_to_weights(model.x)
    #Q[:2,:] = 0.0
    #Q[feat.shape[1]:feat.shape[1]+2,:] = 0.0
    #model.x = model.weights_to_vec(a, h, Q)
    if rtype is not None or len(rtype) > 0:
        for i in range(len(rtype)):
            model.assign_reg_params(rtype[i], reg_params[i])
    print "cost = "
    print model.cost(model.x)
    print ""
    print "gradient = "
    print model.grad(model.x)
    H = model.hess(model.x)
    print ""
    print "Hessian = "
    nice_print(H)

    if cetype is not None and "UV-linear-insert" not in model.cetype and "UV-linear-insert-relaxed" not in model.cetype and use_vars['UVvar']:
        print ""
        print "equality constraints = "
        print model.ceq(model.x)
        print ""
        print "equality constraints Jacobian = "
        print model.ceq_jacobian(model.x)
        print ""
        print "equality constraints Hessian (linear combination weighted by lda) = "
        Hce = model.ceq_hess(model.x, lda_const)
        nice_print(Hce)

    from scipy.optimize import check_grad

    print ""
    print "cost gradient checking (norm difference) = "
    print check_grad(lambda x: model.cost(x.astype(float_dtype)).astype(np.float64), lambda x: model.grad(x.astype(float_dtype)).astype(np.float64), model.x.astype(np.float64), epsilon=np.sqrt(np.finfo(float_dtype).eps))
    #print check_grad(model.cost, model.grad, model.x)

    print ""
    print "cost Hessian checking (norm difference) = "
    for i in range(H.shape[1]):
        print "row " + str(i) + ": " + str(check_grad(lambda x: model.grad(x.astype(float_dtype))[i].astype(np.float64), lambda x: model.hess(x.astype(float_dtype))[i, :].astype(np.float64), model.x.astype(np.float64), epsilon=np.sqrt(np.finfo(float_dtype).eps)))
        #print "row " + str(i) + ": " + str(check_grad(lambda x: model.grad(x)[i], lambda x: model.hess(x)[i, :], model.x))
    
    if cetype is not None and "UV-linear-insert" not in model.cetype and "UV-linear-insert-relaxed" not in model.cetype and use_vars['UVvar']:
        print ""
        print "equality constraints Jacobian checking (norm difference) = "
        for i in range(model.ceq(model.x).size):
            #print "col " + str(i) + ": " + str(check_grad(lambda x: model.ceq(x.astype(float_dtype))[i].astype(np.float64), lambda x: model.ceq_jacobian(x.astype(float_dtype))[:, i].astype(np.float64), model.x.astype(np.float64)))
            print "col " + str(i) + ": " + str(check_grad(lambda x: model.ceq(x)[i], lambda x: model.ceq_jacobian(x)[:, i], model.x))

        print ""
        print "equality constraints Hessian checking (norm difference; linear combination weighted by lda) = "
        for i in range(Hce.shape[1]):
            #print "row " + str(i) + ": " + str(check_grad(lambda x: np.dot(model.ceq_jacobian(x), lda_const)[i], lambda x: model.ceq_hess(x, lda_const)[i, :], model.x))
            print "row " + str(i) + ": " + str(check_grad(lambda x: np.dot(model.ceq_jacobian(x), lda_const)[i], lambda x: model.ceq_hess(x, lda_const)[i, :], model.x))
