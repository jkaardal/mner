import numpy as np

""" constraints.py (module)

    Define constraints classes that can be imposed on the
    hyperparameters. Each class should evaluate the constraints and,
    if Bayesian optimization is desired, include a function that
    builds string expressions for the constraints. To build your own,
    follow along to the examples here.

"""

class BaseCons(object):

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



class SplitSign(BaseCons):
    """ SplitSign (class)

        Constrain multiple nuclear-norm regularization parameters to
        eliminate redundancy when the low-rank MNE model is subjected
        to the linear equality constraints 'UV-linear' or
        'UV-linear-insert'.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the split sign constraints.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class HyperManager
                  from mner.solvers.manager.py) from which the
                  constraints are instantiated.
                  - qualifier: (default="") string that prepends a
                    prefix to the expected keyword arguments.
                  - (qualifier +)csigns: (default=None) numpy array
                    giving the sign relationship between each column
                    of U and V from mner.model.py.
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - (qualifier +)csigns: same as above.

        """
        self.qualifier = parent.get('qualifier', "")
        self.csigns = kwargs.get(self.qualifier + '_csigns', kwargs.get('csigns', None))
        self.initialized = True
        

    def constrain(self, parent=dict(), **kwargs):
        """ Check the constraints to determine if a set of nuclear-norm
            regularization parameters is feasible.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class HyperManager
                  from mner.solvers.manager.py) from which parameters
                  may be passed to define the constraints.
                  - state: dictionary giving the present
                    hyperparameter state (see
                    mner.solvers.manager.py). Must have key
                    "nuclear-norm".
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - (qualifier +)csigns: (default=self.csigns) see
                    class function __init__.

            [returns] feasible                
                feasible: Boolean indicating whether the present
                  nuclear-norm regularization parameters are
                  feasible. If the regularization parameters are
                  feasible, returns True. Otherwise, returns False.

        """
        self.csigns = kwargs.get(self.qualifier + '_csigns', kwargs.get('csigns', self.csigns))
        if parent["state"]["nuclear-norm"].size > 1:
            ind_pos = self.csigns > 0
            ind_neg = self.csigns < 0
            if ind_pos.size > 1:
                val = np.diff(parent["state"]["nuclear-norm"][ind_pos]) < 0
                if np.any(val):
                    # infeasible
                    return False
            if ind_neg.size > 1:
                val = np.diff(parent["state"]["nuclear-norm"][ind_neg]) < 0
                if np.any(val):
                    # infeasible
                    return False
        # feasible
        return True


    def constrain_str(self, parent=dict(), **kwargs):
        """ Build string expressions for the constraints to be used in the
            Bayesian optimization software.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class HyperManager
                  from mner.solvers.manager.py) from which parameters
                  may be passed to define the constraints.
                  - rtype: list of strings of hyperparameter
                    types. Must include "nuclear-norm".
                  - red_dim: integer numpy array of the number of
                    hyperparameters of a given type defined in the
                    reduced hyperparameter space (see
                    mner.solvers.manager.py).
                  - red_index: integer numpy array of the linear span
                    of each hyperparameter type in the reduced
                    hyperparameter space.
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - (qualifier +)csigns: (default=self.csigns) see
                    class function __init__.

            [returns] c_str 
                c_str: list of strings where each element is an
                  expression that, when evaluated, computes a
                  constraint function that is feasible if output is
                  less than or equal to zero.

        """
        # use this for Bayesian Optimization
        c_str = []
        self.csigns = kwargs.get(self.qualifier + '_csigns', kwargs.get('csigns', self.csigns))
        for i, r in enumerate(parent["rtype"]):
            if r == "nuclear-norm" and parent["red_dim"][i] > 1:
                ind_pos = np.where(self.csigns > 0)[0]
                ind_neg = np.where(self.csigns < 0)[0]
                for j in range(ind_pos.size-1):
                    c_str.append('x[:,' + str(parent["red_index"][i]+ind_pos[j]) + '] - x[:,' + str(parent["red_index"][i]+ind_pos[j+1]) + ']')
                for j in range(ind_neg.size-1):
                    c_str.append('x[:,' + str(parent["red_index"][i]+ind_neg[j]) + '] - x[:,' + str(parent["red_index"][i]+ind_neg[j+1]) + ']')
        return c_str


    

class InvariantRtype(BaseCons):
    """ InvariantRtype (class)

        Constrain multiple hyperparameters of a given type to
        eliminate redundancy. This redundancy can be caused by the
        equivalence of the low-rank MNE problem to swapping the
        hyperparameters of a given type.

    """

    def __init__(self, parent=dict(), **kwargs):
        """ Initialize the invariant constraints.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class HyperManager
                  from mner.solvers.manager.py) from which the
                  constraints are instantiated.
                  - qualifier: (default="") string that prepends a
                    prefix to the expected keyword arguments.
                kwargs: Note that keyword arguments take precedence
                  over parent when arguments overlap
                  - (qualifier +)apply_to: (default=[]) list of
                    strings that are subset of rtype that are to be
                    constrained.

        """
        self.qualifier = parent.get("qualifier", "")
        self.apply_to = kwargs.get(self.qualifier + '_apply_to', kwargs.get('apply_to', []))
        if self.apply_to is None:
            self.apply_to = []
        if not isinstance(self.apply_to, list) or isinstance(self.apply_to, tuple):
            self.apply_to = [self.apply_to]

        self.initialized = True

        
    def constrain(self, parent=dict(), **kwargs):
        """ Check the constraints to determine if any of the hyperparameters
            are feasible.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class HyperManager
                  from mner.solvers.manager.py) from which parameters
                  may be passed to define the constraints.
                  - state: dictionary giving the present
                    hyperparameter state (see
                    mner.solvers.manager.py).

            [returns] feasible                
                feasible: Boolean indicating whether the present
                  nuclear-norm regularization parameters are
                  feasible. If the regularization parameters are
                  feasible, returns True. Otherwise, returns False.

        """
        for r in self.apply_to:
            if parent["state"][r].size > 1:
                val = np.diff(parent["state"][r]) < 0
                if np.any(val):
                    # infeasible
                    return False
        # feasible
        return True


    def constrain_str(self, parent=dict(), **kwargs):
        """ Build string expressions for the constraints to be used in the
            Bayesian optimization software.

            [inputs] (parent=dict(), **kwargs)
                parent: object or dictionary composed of the parent
                  instantiation or namespace (e.g. class HyperManager
                  from mner.solvers.manager.py) from which parameters
                  may be passed to define the constraints.
                  - rtype: list of strings of hyperparameter
                    types. Must include "nuclear-norm".
                  - red_dim: integer numpy array of the number of
                    hyperparameters of a given type defined in the
                    reduced hyperparameter space (see
                    mner.solvers.manager.py).
                  - red_index: integer numpy array of the linear span
                    of each hyperparameter type in the reduced
                    hyperparameter space.

            [returns] c_str 
                c_str: list of strings where each element is an
                  expression that, when evaluated, computes a
                  constraint function that is feasible if output is
                  less than or equal to zero.

        """
        # use this for Bayesian Optimization
        c_str = []
        for i, r in enumerate(parent["rtype"]):
            if r in self.apply_to and parent["red_dim"][i] > 1:
                for j in range(parent["red_dim"][i]-1):
                    c_str.append('x[:,' + str(parent["red_index"][i]+j) + '] - x[:,' + str(parent["red_index"][i]+1+j) + ']')
        return c_str


