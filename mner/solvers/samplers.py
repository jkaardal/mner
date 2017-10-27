import numpy as np

""" samplers.py (module)

    Class definitions to ensure feasible sampling of
    hyperparameters. To make customized samplers, follow the form of
    the examples below.

"""

class BaseSampler(object):

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


class InvariantRtypeSampling(BaseSampler):
    """ InvariantRtypeSampling (class)

        Sample a feasible point for invariant hyperparameters. See
        mner.solvers.constraints.InvariantRtype.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the sampler.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class HyperManager
                  from mner.solvers.manager.py) from which the sampler
                  is instantiated.
                  - qualifier: (default="") string that preprends a
                    prefix to the expected keyword arguments.
                  - rtype: (default=[]) list of strings where each
                    string is the name of a hyerparameter type.
                  - red_dim: (default=None) integer numpy array of the
                    number of hyperparameters of a given type defined
                    in the reduced hyperparameter space (see
                    mner.solvers.manager.py).
                  - red_index: (default=None) integer numpy array of
                    the linear span of each hyperparameter type in the
                    reduced hyperparameter space.
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - (qualifier +)apply_to: (default=[]) list of
                    strings that are subset of rtype that are to be
                    constrained.

        """
        self.qualifier = parent.get("qualifier", "")
        self.rtype = parent.get('rtype', [])
        self.red_index = parent.get('red_index', None)
        self.red_dim = parent.get('red_dim', None)

        self.apply_to = kwargs.get(self.qualifier + '_apply_to', kwargs.get('apply_to', []))
        if self.apply_to is None:
            self.apply_to = []
        if not isinstance(self.apply_to, list):
            self.apply_to = [self.apply_to]

        self.initialized = True


    def samp_func(self, space, num_data, **kwargs):
        """ Generate feasible samples.

            [inputs] (bounds, num_data) 

                bounds: iterable of domains for each hyperparameter in
                  the reduced hyperparameter space. Axis=0 corresponds
                  to an element of the hyperparameter space while
                  axis=1 contains the lower and upper
                  bound. E.g. bounds[k][0] is the lower bound of the
                  kth element and bounds[k][1] is the upper bound of
                  the kth element.

                num_data: positive integer for the number of points to
                  genenerate.

        """
        dim = len(space.config_space)
        Z_rand = np.zeros((num_data, dim))
        for k in range(dim):
            if space.config_space[k]['type'] == 'continuous':
                Z_rand[:, k] = np.random.uniform(low=space.config_space[k]['domain'][0], high=space.config_space[k]['domain'][1], size=num_data)

        for i, r in enumerate(self.rtype):
            if r in self.apply_to and self.red_dim[i] > 1:
                Z_rand[:, self.red_index[i]:self.red_index[i+1]] = np.sort(Z_rand[:, self.red_index[i]:self.red_index[i+1]], axis=1)

        return Z_rand



    
class SplitSignSampling(BaseSampler):
    """ SplitSignSampling (class)

        Sample a feasible point for nuclear-norm regularization with
        either the "UV-linear" or "UV-linear-insert" equality
        constraints. See mner.solvers.constraints.SplitSign.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the sampler.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class HyperManager
                  from mner.solvers.manager.py) from which the sampler
                  is instantiated.
                  - qualifier: (default="") string that preprends a
                    prefix to the expected keyword arguments.
                  - rtype: (default=[]) list of strings where each
                    string is the name of a hyerparameter type.
                  - red_dim: (default=None) integer numpy array of the
                    number of hyperparameters of a given type defined
                    in the reduced hyperparameter space (see
                    mner.solvers.manager.py).
                  - red_index: (default=None) integer numpy array of
                    the linear span of each hyperparameter type in the
                    reduced hyperparameter space.
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - (qualifier +)csigns: (default=None) numpy array
                    giving the sign relationship between each column
                    of U and V from mner.model.py.

        """
        self.qualifier = parent.get('qualifier', "")
        self.rtype = parent.get('rtype', [])
        self.red_index = parent.get('red_index', None)
        self.red_dim = parent.get('red_dim', None)
        self.csigns = kwargs.get(self.qualifier + '_csigns', kwargs.get('csigns', None))

    
    def samp_func(self, space, num_data, **kwargs):
        """Generate feasible samples.

            [inputs] (bounds, num_data) 
                bounds: iterable of domains for each hyperparameter in
                  the reduced hyperparameter space. Axis=0 corresponds
                  to an element of the hyperparameter space while
                  axis=1 contains the lower and upper
                  bound. E.g. bounds[k][0] is the lower bound of the
                  kth element and bounds[k][1] is the upper bound of
                  the kth element.
                num_data: positive integer for the number of points to
                  genenerate.
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - (qualifier +)csigns: (default=self.csigns) see
                    definition in __init__.

        """
        self.csigns = kwargs.get(self.qualifier + '_csigns', kwargs.get('csigns', self.csigns))

        dim = len(space.config_space)
        Z_rand = np.zeros((num_data, dim))
        for k in range(dim):
            if space.config_space[k]['type'] == 'continuous':
                Z_rand[:, k] = np.random.uniform(low=space.config_space[k]['domain'][0], high=space.config_space[k]['domain'][1], size=num_data)

        for i, r in enumerate(self.rtype):
            if r == "nuclear-norm" and self.red_dim[i] > 1:
                ind_pos = np.where(self.csigns > 0)[0]
                ind_neg = np.where(self.csigns < 0)[0]
                if ind_pos.size > 1:
                    Z_rand[:, self.red_index[i]+ind_pos] = np.sort(Z_rand[:, self.red_index[i]+ind_pos], axis=1)
                if ind_neg.size > 1:
                    Z_rand[:, self.red_index[i]+ind_neg] = np.sort(Z_rand[:, self.red_index[i]+ind_neg], axis=1)
        
        return Z_rand
    
