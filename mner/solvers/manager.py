import numpy as np

""" manager.py (module)

    This module contains the HyperManager class used for managing the
    hyperparameters of the low-rank MNE problem.

"""

# regularization manager

class HyperManager(object):
    """ HyperManager (class)

        Manage the hyperparameter space by storing an array of
        hyperparameters and the size of each type of hyperparameter. Class
        methods are provided to allow different views of the hyperparameters
        and to manipulate the hyperparameters.

        Note that throughout the description of this class, references
        are made to a "full" hyperparameter space and a "reduced"
        hyperparameter space. The full hyperparameter space includes
        all hyperparameters including dependent hyperparameters. In
        this case, dependent hyperparameters are those whose value is
        a function of another hyperparameter; e.g. eps_2 = 2.0*eps_1
        would indicate that eps_2 is dependent on eps_1 (or
        vice-versa). In the reduced hyperparameter space, all
        hyperparameters are set independently; e.g. since eps_1 and
        eps_2 are dependent, only one of the two is chosen as the
        independent hyperparameter and the other is removed from the
        reduced hyperparameter space.

    """

    def __init__(self, qualifier="", parent=dict(), **kwargs):
        """ Initialize the hyperparamater manager.

            [inputs] (qualifier="", parent=dict(), **kwargs)
                qualifier: (optional) string that preppends a prefix
                  to the expected keyword arguments (default "").
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. a class from
                  solvers.py) from which the manager is instantiated.
                  - train_model: (default=None) low-rank MNE model
                    evaluated on the training set
                  - (qualifier + )rtype: (default=None) list of
                    strings indicating the type of regularization
                    penalties applied to the training model.
                  - (qualifier + )hypergrid: (default=None) list
                    containing grids of hyperparameters (e.g. for
                    doing a grid search). The grid is formatted as
                    follows:

                        Each element of hypergrid corresponds to
                        hyperparameter in the element of rtype with
                        the same index. Within each element of the
                        list, the grid can be defined in three
                        different ways...
                        - Case 1: for a single hyperparameter of a
                          given type, the element is a 1D numpy array
                          defining the grid. E.g. hypergrid =
                          [np.arange(0.0, 1.0, 0.1)].
                        - Case 2: for multiple hyperparameters of a
                          given type, the element is a 2D numpy array
                          where each column corresponds to one of the
                          hyperparameters (in order) and each row
                          corresponds to a point on the grid. This
                          grid is still ultimately one-dimenionsal
                          because the grid is traversed row-by-row and
                          does not try enumerate combinatorial points
                          on the grid. E.g. hypergrid =
                          [np.array([[0.0, 0.1, 0.2],
                          [1.0. 2.0. 3.0]])] for two hyperparameters
                          of a given type.
                        - Case 3: for multiple hyperparameters of a
                          given type where feasible combinations of
                          hyperparameters are to be enumerated, the
                          element is a list of one-dimensional numpy
                          arrays where each element of the list
                          corresponds to a hyperparameter (in order)
                          and each one-dimenional numpy array is a
                          grid. E.g. hypergrid = [[np.arange(0.0, 1.0,
                          0.1)]*5] for five independent
                          hyperparameters of a given type.

                  - (qualifier + )domain: (default=None) domain of
                    each hyperparameter. The format is the same as
                    hypergrid except each "grid" has only a length of
                    two (for the lower and upper bounds).
                  - (qualifier + )deps: (default=None) dictionary of
                    functions where each key is a string from rtype
                    and each value is a function that constructs the
                    state of the given hyperparameter type in a
                    greater dimensional space.
                  - (qualifier + )cons: (default=None) list of
                    inequality constraints to be applied to the
                    hyperparameters. Each element of the list is a
                    class that imposes the constraints (see
                    mner.solvers.constraints.py for more information).
                  - (qualifier + )sampler:
                    (default=HyperManager.default_sampler) function
                    that returns feasible hyperparameter vectors
                  - float_dtype: (default=np.float64) floating-point
                    data type
                kwargs: Note that keyword arguments take precedence
                over parent when arguments overlap
                  - same variables as parent.

        """
        # get relevant information
        self.qualifier = qualifier
        self.train_model = kwargs.get('train_model', parent.get('train_model', None))
        self.rtype = kwargs.get(qualifier + '_rtype', kwargs.get('rtype', parent.get('rtype', None)))
        self.grid = kwargs.get(qualifier + '_hypergrid', kwargs.get('hypergrid', parent.get('hypergrid', None)))
        self.domain = kwargs.get(qualifier + '_domain', kwargs.get('domain', parent.get('domain', None)))
        self.deps = kwargs.get(qualifier + '_deps', kwargs.get('deps', parent.get('deps', dict())))
        self.cons = kwargs.get(qualifier + '_cons', kwargs.get('cons', parent.get('cons', None)))
        # for consistency, make cons a list
        if isinstance(self.cons, list) or isinstance(self.cons, tuple) and len(self.cons) == 0:
            self.cons = None
        if self.cons is not None:
            self.cons = [self.cons]
        # get sampler function
        self.sampler = kwargs.get(qualifier + '_sampler', kwargs.get('sampler', parent.get('sampler', self.default_sampler)))
        
        # initialize unset class variables
        self.index = None
        self.red_index = None
        self.dim = None
        self.red_dim = None
        self.length = None

        # initialize the state dict
        self.state = dict()

        # initialize the domain and constraint dicts
        self.domain_f = None
        self.cons_f = None
        
        # get the floating-point data type
        self.float_dtype = kwargs.get('float_dtype', parent.get('float_dtype', np.float64))

        # setup the manager
        self.setup(**kwargs)

        self.initialized = True


    class default_sampler:
        """ The default sampler subclass tells the Bayesian optimization
            software to use its domain sampler for initializing the
            acquisition function optimization.

        """

        def __init__(self, parent=dict(), **kwargs):
            """ Initialize the default sampler.

                [inputs] (parent=dict(), **kwargs)
                    parent: does nothing here.

                [returns] None

            """
            self.samp_func = None
        

    def setup(self, **kwargs):
        """ Set-up the hyperparameter manager. The set-up procedure
            automatically determines the size of each type of
            hyperparameter subspace and determines the indices that
            span each hyperparameter type in both the full
            hyperparameter space and the reduced hyperparameter
            space. The constraint and sampler classes are
            instantiated.

            [inputs] (**kwargs)

        """
        if self.rtype is not None:
            self.build_dim(**kwargs)
            self.build_red_dim(**kwargs)
            self.build_length(**kwargs)
            self.build_index(**kwargs)
            self.build_red_index(**kwargs)

            if self.cons is not None:
                # initialize constraints
                for i in range(len(self.cons)):
                    self.cons[i] = self.cons[i](self, **kwargs)

            if self.sampler is not None:
                # initialize sampling function
                self.sampler = self.sampler(self, **kwargs)
                        

    def check_feasibility(self, **kwargs):
        """ Check the constraints to see if a given hyperparameter state is
            feasible.

            [inputs] (**kwargs)

            [returns] feasible
                 feasible: Boolean that is True if the current state
                   is feasible and False if the current state is
                   infeasible.

        """
        feasible = True
        if self.cons is not None:
            for i in range(len(self.cons)):
                if not self.cons[i].constrain(self, **kwargs):
                    # if infeasbile, break
                    feasible = False
                    break
        return feasible


    def build_index(self, **kwargs):
        """ Build an index that holds the span of the elements that make up
            the hyperparameters of each type in rtype as they appear
            in a full hyperparameter vector.

            [inputs] (**kwargs)

            [returns] index 
                index: integer numpy array of indices that correspond
                  to the index that follows the last element of a
                  hyperparameter vector of each type in rtype (in
                  order).

        """
        idx = 0
        self.index = np.zeros((1, 1), dtype=np.uint32)
        for i, r in enumerate(self.rtype):
            idx += self.get_dim(r, i, **kwargs)
            self.index = np.concatenate([self.index, np.array([idx]).reshape((1, 1))], axis=0)
        self.index = self.index.ravel()
        return self.index
            

    def build_red_index(self, **kwargs):
        """ Build an index that holds the span of the elements that make up
            the hyperparameters of each type in rtype as they appear
            in a reduced hyperparameter vector.

            [inputs] (**kwargs)

            [returns] red_index
                red_index: integer numpy array of indices that
                  correspond to the index that follows the last
                  element of a reduced hyperparameter vector of each
                  type in rtype (in order).

        """
        idx = 0
        self.red_index = np.zeros((1, 1), dtype=np.uint32)
        for i, r in enumerate(self.rtype):
            idx += self.get_red_dim(r, i, **kwargs)
            self.red_index = np.concatenate([self.red_index, np.array([idx]).reshape((1, 1))], axis=0)
        self.red_index = self.red_index.ravel()
        return self.red_index


    def build_dim(self, **kwargs):
        """ Create a one-dimensional numpy array that holds the number of
            hyperparameters of each type in rtype (in order) in the
            full hyperparameter space.

            [inputs] (**kwargs)

            [returns] dim
                dim: integer numpy array of the number of
                  hyperparameters of each type in the same order as
                  rtype.

        """
        self.dim = np.zeros((0, 1), dtype=np.uint32)
        for i, r in enumerate(self.rtype):
            self.dim = np.concatenate([self.dim, np.array([self.get_dim(r, i, **kwargs)]).reshape((1, 1))], axis=0)
        self.dim = self.dim.ravel()
        return self.dim


    def build_red_dim(self, **kwargs):
        """ Create a one-dimensional numpy array that holds the number of
            hyperparameters of each type in rtype (in order) in the
            reduced hyperparameter space.

            [inputs] (**kwargs)

            [returns] red_dim
                dim: integer numpy array of the number of independent
                  hyperparameters of each type in the same order as
                  rtype.

        """
        self.red_dim = np.zeros((0, 1), dtype=np.uint32)
        for i, r in enumerate(self.rtype):
            self.red_dim = np.concatenate([self.red_dim, np.array([self.get_red_dim(r, i, **kwargs)]).reshape((1, 1))], axis=0)
        self.red_dim = self.red_dim.ravel()
        return self.red_dim


    def build_length(self, **kwargs):
        """ Build an integer numpy array of hypergrid lengths for each
            hyperparameter in the reduced hyperparameter space.

            [inputs] (**kwargs)

            [returns] length
                length: integer numpy array holding the lengths of the
                  hypergrid for each element of the reduced
                  hyperparameter space.

        """
        self.length = np.zeros((0, 1), dtype=np.uint32)
        for i, r in enumerate(self.rtype):
            self.length = np.concatenate([self.length, self.get_length(r, i, **kwargs)], axis=0)
        self.length = self.length.ravel()
        return self.length


    def build_state_from_grid(self, indices=None, **kwargs):
        """ Given indices for each element of the reduced hyperparameter space,
            build the hyperparameter state dictionary from the
            hypergrid.

            [inputs] (indices=None, **kwargs) 
                indices: iterable container where each element
                  corresponds to an element of the reduced
                  hyperparameter space.

            [returns] state
                state: dictionary where each key is a string from
                  rtype with value equal to a numpy array of the full
                  hyperparameter space of the given type.

        """
        for i, r in enumerate(self.rtype):
            if self.grid is not None and self.grid[i] is not None:
                if isinstance(indices, dict):
                    self.state[r] = self.get_grid_state(r, i, indices[r][self.red_index[i]:self.red_index[i+1]], **kwargs)
                else:
                    self.state[r] = self.get_grid_state(r, i, indices[self.red_index[i]:self.red_index[i+1]], **kwargs)
        return self.state

                    
    def build_state_from_vector(self, v=None, **kwargs):
        """ Given a vector of hyperparameters (defined in the full
            hyperparameter space), build the state dictionary.

            [inputs] (v=None, **kwargs)
                v: numpy array of hyperparameters defined in the full
                  hyperparameter space.

            [returns] state
                state: dictionary where each key is a string from
                  rtype with value equal to a numpy array of the full
                  hyperparameter space of a given type.

        """
        for i, r in enumerate(self.rtype):
            self.state[r] = v[self.index[i]:self.index[i+1]]


    def build_state_from_red_vector(self, v=None, **kwargs):
        """ Given a vector of hyperparameters defined in the reduced
            hyperparameter space, build the state dictionary.

            [inputs] (v=None, **kwargs)
                v: numpy array of hyperparameters defined in the
                  reduced hyperparameter space.

            [returns] state
                state: dictionary where each key is a string from
                  rtype with value equal to a numpy array of the full
                  hyperparameter space of a given type.

        """
        for i, r in enumerate(self.rtype):
            self.state[r] = v[self.red_index[i]:self.red_index[i+1]]
            if self.dim[i] != self.red_dim[i]:
                if r in self.deps:
                    self.state[r] = self.deps[r](parent=self, **kwargs).ravel()
                else:
                    self.state[r] = np.tile(self.state[r].reshape((1, 1)), (self.dim[i], 1)).ravel()
        return self.state


    def build_vector_from_state(self, **kwargs):
        """ Build a hyperparameter vector defined in the full hyperparameter
            space from the hyperparameter state dictionary.

            [inputs] (**kwargs)

            [returns] v
                v: numpy array of hyperparameters defined in the full
                  hyperparameter space.

        """
        v = np.zeros((self.index[-1],))
        for i, r in enumerate(self.rtype):
            v[self.index[i]:self.index[i+1]] = np.copy(self.state[r])
        return v


    def build_red_vector_from_state(self, **kwargs):
        """ Build a hyperparameter vector defined in the reduced
            hyperparameter space from the hyperparameter state
            dictionary.

            [inputs] (**kwargs)

            [returns] v
                v: numpy array of hyperparameters defined in the
                  reduced hyperparameter space.

        """
        v = np.zeros((self.red_index[-1],))
        for i, r in enumerate(self.rtype):
            if self.dim[i] != self.red_dim[i]:
                v[self.red_index[i]:self.red_index[i+1]] = self.state[r][0]
            else:
                v[self.red_index[i]:self.red_index[i+1]] = self.state[r][self.index[i]:self.index[i+1]]
        return v


    def build_red_vector_from_vector(self, v=None, **kwargs):
        """ Build a hyperparameter vector defined in the reduced
            hyperparameter space from a hyperparameter vector defined
            in the full hyperparameter space.

            [inputs] (v=None, **kwargs)
                v: numpy array of hyperparameters defined in the full
                  hyperparameter space.

            [returns] v_red
                v_red: numpy array of hyperparameters defined in the
                  reduced hyperparameter space.

        """

        self.build_state_from_vector(v, **kwargs)
        return self.build_red_vector_from_state(**kwargs)


    def build_vector_from_red_vector(self, v=None, **kwargs):
        """ Build a hyperparameter vector defined in the full hyperparameter
            space from a hyperparameter vector defined in the reduced
            hyperparameter space.

            [inputs] (v=None, **kwargs)
                v: numpy array of hyperparameters defined in the
                  reduced hyperparameter space.

            [returns] v_full
                v_full: numpy array of hyperparameters defined in the
                  full hyperparameter space.

        """
        self.build_state_from_red_vector(v, **kwargs)
        return self.build_vector_from_state(**kwargs)
    
    
    def update_model(self, **kwargs):
        """ Update the hyperparameters in the training model.

            [inputs] (**kwargs)

        """
        for i, r in enumerate(self.rtype):
            self.train_model.assign_reg_params(r, self.state[r], **kwargs)
        

    def get_dim(self, r=None, idx=None, **kwargs):
        """ Get the number of dimensions in the full hyperparameter space of
            type r (from rtype).

            [inputs] (r=None, idx=None, **kwargs)
                r: (optional) string signaling the hyperparameter type
                  to get.
                idx: (optional) integer index of rtype signaling the
                  hyperparameter type to get (default None).

            [returns] rdim 
                rdim: the number of hyperparameters contributing to
                  the dimensionality of the full hyperparameter space
                  of type r (or idx).

        """
        if idx is None and r is not None:
            idx = np.where(self.rtype == r)[0]
        if r is None and idx is not None:
            r = self.rtype[idx]
        if r in self.rtype:
            if self.grid is not None and self.grid[idx] is not None:
                if isinstance(self.grid[idx], list) or isinstance(self.grid[idx], tuple):
                    return len(self.grid[idx])
                elif self.grid[idx].shape[0] == self.grid[idx].size:
                    return 1
                else:
                    return self.grid[idx].shape[1]
            elif self.domain is not None and self.domain[idx] is not None:
                if isinstance(self.domain[idx], list) or isinstance(self.domain[idx], tuple):
                    return len(self.domain[idx])
                elif self.domain[idx].shape[0] == self.domain[idx].size:
                    return 1
                else:
                    return self.domain[idx].shape[1]

            return 0

        return None


    def get_red_dim(self, r=None, idx=None, **kwargs):
        """ Get the number of dimensions in the reduced hyperparameter space
            of type r (from rtype).

            [inputs] (r=None, idx=None, **kwargs)
                r: (optional) string signaling the hyperparameter type
                  to get.
                idx: (optional) integer index of rtype signaling the
                  hyperparameter type to get.

            [returns] rdim 
                rdim: the number of hyperparameters contributing to
                  the dimensionality of the reduced hyperparameter
                  space of type r (or idx).

        """
        if idx is None and r is not None:
            idx = np.where(self.rtype == r)[0]
        if r is None and idx is not None:
            r = self.rtype[idx]
        if r in self.rtype:
            if self.grid is not None and self.grid[idx] is not None:
                if isinstance(self.grid[idx], list) or isinstance(self.grid[idx], tuple):
                    return len(self.grid[idx])
                else:
                    return 1
            elif self.domain is not None and self.domain[idx] is not None:
                if isinstance(self.domain[idx], list) or isinstance(self.domain[idx], tuple):
                    return len(self.domain[idx])
                else:
                    return 1

            return 0

        return None


    def get_length(self, r=None, idx=None, **kwargs):
        """ Get length of the grid(s) for each hyperparameter of type r from
            rtype.

            [inputs] (r=None, idx=None, **kwargs)
                r: (optional) string signaling the hyperparameter type
                  to get.
                idx: (optional) integer index of rtype signaling the
                  hyperparameter type to get.

            [returns] rlen
                rlen: returns a numpy array where each element
                  corresponds to the length of the hypergrid in the
                  reduced hyperparameter space.

        """
        if idx is None and r is not None:
            idx = np.where(self.rtype == r)[0]
        if r is None and idx is not None:
            r = self.rtype[idx]
        if r in self.rtype:
            if self.grid is not None and self.grid[idx] is not None:
                if isinstance(self.grid[idx], list) or isinstance(self.grid[idx], tuple):
                    x = np.array([x.size for x in self.grid[idx]])
                    return x.reshape((x.size, 1))
                else:
                    x = np.array([self.grid[idx].shape[0]])
                    return x.reshape((1, 1))
            elif self.domain is not None and self.domain[idx] is not None:
                if isinstance(self.domain[idx], list) or isinstance(self.domain[idx], tuple):
                    return np.zeros((1, 1))
                else:
                    return np.zeros((1, 1))

            return np.zeros((1, 1))

        return None


    def get_grid_state(self, r=None, idx=None, indices=None, **kwargs):
        """ Get grid elements of type r in the full hyperparameter space.

            [inputs] (r=None, idx=None, indices=None, **kwargs)
                r: (optional) string signaling the hyperparameter type
                  to get.
                idx: (optional) integer index of rtype signaling the
                  hyperparameter type to get.
                indices: (optional) integer numpy array of an index or
                  indices on the hypergrid for each hyperparameter of
                  type r. If the hyperparameters are dependent, then
                  there is only one index. If the hyperparameters are
                  independent, then size of indices should be equal to
                  the number of hyperparameters of type r.
            [returns] rstate
                rstate: numpy array of hyperparameters of type r (or
                  idx) defined in the full hyperparameter space.

        """
        if idx is None and r is not None:
            idx = np.where(self.rtype == r)[0]
        if r is None and idx is not None:
            r = self.rtype[idx]
        if r in self.rtype:
            if isinstance(self.grid[idx], list) or isinstance(self.grid[idx], tuple):
                composite = np.zeros((len(self.grid[idx]),))
                for i in range(len(self.grid[idx])):
                    composite[i] = self.grid[idx][i][indices[i]]
                return composite.ravel()
            else:
                if self.grid[idx].shape[0] == self.grid[idx].size:
                    return np.array([self.grid[idx][indices]]).ravel()
                else:
                    x = np.array([self.grid[idx][indices,:]])
                    return x.ravel()
        
        return None


    def format_domain(self, **kwargs):
        """ Format the domain as a list of dictionaries (assumes
            hyperparameters are continuous) for use in the Bayesian
            optimization software.

            [inputs] (**kwargs)

            [returns] domain_f
                domain_f: list of dictionaries where each element is
                  the domain of a hyperparameter defined in the
                  reduced hyperparameter space.

        """
        idx = 0
        self.domain_f = []
        if self.domain is None:
            self.domain_f = None
            return self.domain_f
        for i in range(len(self.domain)):
            if isinstance(self.domain[i], dict):
                self.domain_f.append(self.domain[i])
                self.domain_f[idx]['name'] = 'hyperparam_' + str(idx)
                #if 'colsize' not in self.domain_f[idx]:
                #    self.domain_f[idx]['colsize'] = 1
                idx += 1
            elif isinstance(self.domain[i], list) or isinstance(self.domain[i], tuple):
                tmp = []
                for j in range(len(self.domain[i])):
                    tmp.append({'name': 'hyperparam_' + str(idx), 'type': 'continuous', 'domain': (self.domain[i][j][0], self.domain[i][j][-1])})
                    #tmp.append({'name': 'hyperparam_' + str(idx), 'type': 'continuous', 'domain': (self.domain[i][j][0], self.domain[i][j][-1]), 'colsize': 1})
                    idx += 1
                self.domain_f.extend(tmp)
            else:
                if self.domain[i].shape[1] > 0:
                    self.domain_f.append({'name': 'hyperparam_' + str(idx), 'type': 'continuous', 'domain': (self.domain[i][0, 0], self.domain[i][-1, 0])})
                    #self.domain_f.append({'name': 'hyperparam_' + str(idx), 'type': 'continuous', 'domain': (self.domain[i][0, 0], self.domain[i][-1, 0]), 'colsize': self.domain[i].shape[1]})
                else:
                    self.domain_f.append({'name': 'hyperparam_' + str(idx), 'type': 'continuous', 'domain': (self.domain[i][0], self.domain[i][-1])})
                    #self.domain_f.append({'name': 'hyperparam_' + str(idx), 'type': 'continuous', 'domain': (self.domain[i][0], self.domain[i][-1]), 'colsize': 1})
                idx += 1
        return self.domain_f


    def format_cons(self, **kwargs):
        """ Format the constraints as a list of dictionaries for use in the
            Bayesian optimization software.

            [inputs] (**kwargs)

            [returns] cons_f
                cons_f: list of dictionaries where each element
                  corresponds to a constraint defined in the reduced
                  hyperparameter space.

        """
        idx = 0
        self.cons_f = []
        if self.cons is None:
            self.cons_f = None
            return self.cons_f
        for i in range(len(self.cons)):
            c_str = self.cons[i].constrain_str(self, **kwargs)
            for c in c_str:
                self.cons_f.append({'name': 'hypercon_' + str(idx), 'constrain': c})
                idx += 1
        return self.cons_f

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
