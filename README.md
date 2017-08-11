# Low-rank second-order maximum noise entropy (MNE) modeling 

Low-rank second-order MNE models can be constructed and optimized using the `mner` python package. Second-order MNE models have conditional probability distribution of the form

&nbsp;&nbsp;&nbsp;&nbsp; _P_(_y_=1|<b>s</b>) = 1/[1 + exp(-_a_ - <b>h</b><sup>T</sup><b>s</b> - <b>s</b><sup>T</sup><b>J</b><b>s</b>)]

where _y_ is a reponse in the domain [0, 1], <b>s</b> is a _D_-dimensional feature vector (independent variables), and the unknown weights are the scalar _a_, vector <b>h</b>, and matrix <b>J</b>. Low-rank second-order MNE models decrease the number of weights in the model by substituting <b>J</b> with the bilinear factorization <b>J</b> = <b>UV</b><sup>T</sup> where <b>U</b> and <b>V</b> are _D_ by _r_ matrices of maximum rank _r_.

## Installation

After downloading and unzipping the source code, find the directory containing the setup.py file. Then enter the command:

    python setup.py install

into your terminal. If `setuptools` is installed, this should also install the dependencies. If not, the dependencies are:

    numpy >= 1.7.1
    scipy >= 0.11
    theano >= 0.8.2
    
and the git repositories located at

    http://github.com/jkaardal/GPyOpt/
    http://github.com/jkaardal/pyipm/
    
## Creating and solving problems

See the doc strings in the source code for more information about how to construct and solve low-rank second-order MNE problems.
