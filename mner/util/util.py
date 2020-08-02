import os
import numpy as np
import h5py
import sys

# expand execution path
def expand_execution_path(paths):
    if 'expanded_execution_path' in paths:
        if isinstance(paths, str):
            sys.path.insert(0, os.path.expanduser(paths))
        else:
            for p in paths:
                sys.path.insert(0, os.path.expanduser(p))

                
# generic dataset functions
def load_response(resp_path, format_str, input_dtype, output_dtype):
    # load either a text or binary data response file
    
    if format_str.startswith("t"):
        with open(os.path.expanduser(resp_path), "r") as f:
            y = np.loadtxt(f, dtype=input_dtype)
    else:
        with open(os.path.expanduser(resp_path), "rb") as f:
            y = np.fromfile(f, dtype=input_dtype)

    nsamp = y.size
    ymax = np.max(y)

    y = np.reshape(y, (nsamp,)).astype(output_dtype)

    y = y/ymax

    return y, nsamp, ymax


def load_features(feat_path, format_str, input_dtype, output_dtype, nsamp, feat_major=False):
    # load either a text or binary data file containing the feature samples

    if format_str.startswith("t"):
        with open(os.path.expanduser(feat_path), "r") as f:
            s = np.loadtxt(f, dtype=input_dtype)
    else:
        with open(os.path.expanduser(feat_path), "rb") as f:
            s = np.fromfile(f, dtype=input_dtype)

    ndim = s.size/nsamp

    if feat_major:
        s = np.copy(np.reshape(s, (ndim, nsamp)).T).astype(output_dtype)
    else:
        s = np.reshape(s, (nsamp, ndim)).astype(output_dtype)

    return s, ndim


def zscore_features(s):
    # z-score feature normalization

    nsamp = s.shape[0]
    s_avg = np.mean(s, axis=0)
    s_std = np.std(s, axis=0)
    s -= np.tile(s_avg, (nsamp, 1))
    s /= np.tile(s_std, (nsamp, 1))

    return s, s_avg, s_std


def generate_dataset_logical_indices(train_fraction, cv_fraction, nsamp, njack=1):
    # generate masks corresponding to the training, cross-validation, and test sets
    
    nshift = int(nsamp/njack)
    ntrain = int(nsamp*train_fraction)
    if train_fraction + cv_fraction == 1.0:
        ncv = nsamp - ntrain
        ntest = 0
    else:
        ncv = int(nsamp*cv_fraction)
        ntest = nsamp - ncv - ntrain

    trainset = np.zeros((nsamp,), dtype=bool)
    trainset[:ntrain] = True

    cvset = np.zeros((nsamp,), dtype=bool)
    cvset[ntrain:ntrain+ncv] = True

    testset = np.zeros((nsamp,), dtype=bool)
    if ntest != 0:
        testset[ntrain+ncv:] = True

    nshift = int(nsamp/njack)

    return trainset, cvset, testset, nshift


def roll_dataset_logical_indices(trainset, cvset, testset, nshift, djack):

    trainset = np.roll(trainset, djack*nshift)
    cvset = np.roll(cvset, djack*nshift)
    testset = np.roll(testset, djack*nshift)

    return trainset, cvset, testset


def convert_dataset_logical_indices_to_array_indices(trainset, cvset, testset):

    trainInd = np.where(trainset)[0][:]
    cvInd = np.where(cvset)[0][:]
    testInd = np.where(testset)[0][:]

    trainInd = np.reshape(trainInd, (trainInd.size,))
    cvInd = np.reshape(cvInd, (cvInd.size,))
    testInd = np.reshape(testInd, (testInd.size,))

    return trainInd, cvInd, testInd


# MNEr weight conversions
def weights_to_vec(a=None, h=None, U=None, V=None, Q=None, **kwargs):
    # transform weight matrices into a weight vector
    
    if a is None:
        return np.array([])
    assert a.size == 1

    if h is not None:
        ndim = h.size
    if U is not None:
        if "csigns" not in kwargs:
            assert V is not None
        else:
            csigns = kwargs.get("csigns")
            V = np.dot(U, np.diag(csigns.ravel()))
        if h is not None:
            assert U.shape[0] == ndim
        else:
            ndim = U.shape[0]
        rank = U.shape[1]
        assert (V.shape[0] == ndim) and (V.shape[1] == rank)
        Q = np.concatenate([U, V], axis=0)
    elif Q is not None:
        if h is not None:
            assert Q.shape[0] == 2*ndim
        else:
            ndim = Q.shape[0]/2
        rank = Q.shape[1]            
    else:
        Q = None

    if (h is not None) and (Q is not None):
        return np.concatenate([a.reshape((1,)), h.reshape((h.size,)), Q.T.reshape((Q.size,))])
    elif (h is not None):
        return np.concatenate([a.reshape((1,)), h.reshape((h.size,))])
    elif (Q is not None):
        return np.concatenate([a.reshape((1,)), Q.T.reshape((Q.size,))])
    else:
        return a.reshape((1,))


def vec_to_weights(x, ndim, rank, **kwargs):
    # transform weight vector to weight matrices
    
    if x.size == 1:
        a = np.copy(x).reshape((1,))
        h = None
        U = None
        V = None
    elif x.size == (1+ndim):
        a = np.copy(x[0]).reshape((1,))
        h = np.copy(x[1:ndim+1]).reshape((ndim,))
        U = None
        V = None
    elif x.size == (1+ndim+ndim*rank):
        a = np.copy(x[0]).reshape((1,))
        h = np.copy(x[1:ndim+1]).reshape((ndim,))
        U = np.copy(x[1+ndim:1+(1+rank)*ndim].reshape((rank, ndim)).T)
        if "csigns" not in kwargs:
            V = None
        else:
            csigns = kwargs.get("csigns")
            V = np.dot(U, np.diag(csigns.ravel())) 
    elif x.size == (1+ndim+2*ndim*rank):
        a = np.copy(x[0]).reshape((1,))
        h = np.copy(x[1:ndim+1]).reshape((ndim,))
        Q = np.copy(x[1+ndim:1+(1+2*rank)*ndim].reshape((rank, 2*ndim)).T)
        U = Q[:ndim,:]
        V = Q[ndim:,:]
    else:
        assert x.size == 0
        a = None
        h = None
        U = None
        V = None

    return a, h, U, V


# MNEr block coordinate descent results I/O
def load_blk_weights(rank, ndim, jack=1, path=os.getcwd(), prefix="", suffix="", in_float_dtype=np.float64, 
                     out_float_dtype=np.float64, custom_files=None, order=None):
    # load block results from file
    
    U = np.zeros((ndim, rank)).astype(out_float_dtype)
    V = np.zeros((ndim, rank)).astype(out_float_dtype)
    if custom_files is None or (custom_files is not None and isinstance(custom_files, list)):
        if order is None:
            order = range(rank)
        for r in order:
            if custom_files is not None:
                in_file = custom_files[r]
            else:
                in_file = ""
                if len(prefix):
                    in_file += prefix + "_"
                in_file += "r" + str(rank) + "_b" + str(r+1) + "_j" + str(jack)
                if len(suffix):
                    in_file += "_" + suffix
                in_file += ".prm"
                in_file = os.path.join(path, in_file)
            with open(os.path.expanduser(in_file), "r") as f:
                x = np.fromfile(f, dtype=in_float_dtype)

            a = x[0].reshape(1,).astype(out_float_dtype)
            h = x[1:ndim+1].astype(out_float_dtype)
            U[:,r] = x[ndim+1:2*ndim+1].astype(out_float_dtype)
            V[:,r] = x[2*ndim+1:].astype(out_float_dtype)
    else:
        raise Exception("'custom_files' must be a list of strings.")

    return a, h, U, V


def save_blk_weights(xblk, r, rank, jack=1, path=os.getcwd(), prefix="", suffix="", out_float_dtype=np.float64, 
                     custom_file=None):
    # save block results to file

    if custom_file is None:
        out_file = ""
        if len(prefix):
            out_file += prefix + "_"
        out_file += "r" + str(rank) + "_b" + str(r+1) + "_j" + str(jack)
        if len(suffix):
            out_file += "_" + suffix
        out_file += ".prm"
        out_file = os.path.join(path, out_file)
    elif isinstance(custom_file, str):
        out_file = custom_file
    else:
        raise Exception("'custom_file' must be a string")

    with open(os.path.expanduser(out_file), "w") as f:
        xblk.astype(out_float_dtype).tofile(f)


# MNEr block coordinate descent checkpointing
def save_blk_checkpoint(xblk, r, rank, iter_num, jack=1, path=os.getcwd(), prefix="", suffix="", 
                        out_float_dtype=np.float64, custom_file=None):
    # save checkpoint

    chkpnt_file = ""
    if len(prefix):
        chkpnt_file += prefix + "_"
    chkpnt_file += "r" + str(rank) + "_j" + str(jack)
    if len(suffix):
        chkpnt += "_" + suffix
    chkpnt_file += ".chk"
    chkpnt_file = os.path.join(path, chkpnt_file)

    if os.path.isfile(os.path.expanduser(chkpnt_file)):
        chkpnt = []
        with open(os.path.expanduser(chkpnt_file), "r") as f:
            for line in f:
                chkpnt.append(line.strip().split(','))
        if isinstance(chkpnt[-1], list):
            chkpnt = chkpnt[-1]
        chkpnt = [str(x) for x in chkpnt]
        if len(chkpnt) == rank+1:
            chkpnt = [str(iter_num), str(r)]
        else:
            chkpnt.append(str(r))
    else:
        chkpnt = [str(iter_num), str(r)]

    save_blk_weights(xblk, r, rank, jack, path, prefix, suffix, out_float_dtype, custom_file)

    with open(os.path.expanduser(chkpnt_file), "w") as f:
        f.write(",".join(chkpnt))
    
    return chkpnt


def load_blk_checkpoint(rank, ndim, jack=1, path=os.getcwd(), prefix="", suffix="", in_float_dtype=np.float64, 
                        out_float_dtype=np.float64, custom_files=None):
    # load checkpointed data

    chkpnt_file = ""
    if len(prefix):
        chkpnt_file += prefix + "_"
    chkpnt_file += "r" + str(rank) + "_j" + str(jack)
    if len(suffix):
        chkpnt_file += "_" + suffix
    chkpnt_file += ".chk"
    chkpnt_file = os.path.join(path, chkpnt_file)
    
    chkpnt = []
    x = None
    if os.path.isfile(os.path.expanduser(chkpnt_file)):
        with open(os.path.expanduser(chkpnt_file), "r") as f:
            for line in f:
                chkpnt.append(line.strip().split(','))
        if isinstance(chkpnt[-1], list):
            chkpnt = chkpnt[-1]
        chkpnt = [int(x) for x in chkpnt]
        if len(chkpnt) > 1:
            if chkpnt[0] > 1:
                remainder = list(range(rank))
                remainder = set(remainder) - set(chkpnt[1:])
                order = sorted(list(remainder))
                order.extend(chkpnt[1:])
            else:
                order = chkpnt[1:]
            a, h, U, V = load_blk_weights(rank, ndim, jack, path, prefix, suffix, in_float_dtype, out_float_dtype, 
                                          custom_files, order)
            x = weights_to_vec(a, h, U, V)

    return x, chkpnt
    

def clear_checkpoint(rank, ndim, jack=1, path=os.getcwd(), prefix="", suffix="", custom_files=None):
    # clear checkpoint data

    chkpnt_file = ""
    if len(prefix):
        chkpnt_file += prefix + "_"
    chkpnt_file += "r" + str(rank) + "_j" + str(jack)
    if len(suffix):
        chkpnt_file += "_" + suffix
    chkpnt_file += ".chk"
    chkpnt_file = os.path.join(path, chkpnt_file)
    
    if os.path.isfile(os.path.expanduser(chkpnt_file)):
        chkpnt = []
        with open(os.path.expanduser(chkpnt_file), "r") as f:
            for line in f:
                chkpnt.append(line.strip().split(','))
        if isinstance(chkpnt[-1], list):
            chkpnt = chkpnt[-1]
        chkpnt = [int(x) for x in chkpnt]
        if len(chkpnt) > 1:
            if chkpnt[0] > 1:
                order = range(1, rank+1)
            else:
                order = chkpnt[1:]
            if custom_files is None or (custom_files is not None and isinstance(custom_files, list)):
                for r in order:
                    if custom_files is not None:
                        del_file = custom_files[r-1]
                    else:
                        del_file = ""
                        if len(prefix):
                            del_file += prefix + "_"
                        del_file += "r" + str(rank) + "_j" + str(jack)
                        if len(suffix):
                            del_file += "_" + suffix
                        del_file += ".prm"
                        del_file = os.path.join(path, del_file)
        
                    try:
                        os.remove(os.path.expanduser(del_file))
                    except OSError:
                        pass
            else:
                raise Exception("'custom_files' must be a list of strings.")
        try:
            os.remove(os.path.expanduser(chkpnt_file))
        except OSError:
            pass

    return []


# MNEr results I/O
def save_weights(x, rank, jack=1, path=os.getcwd(), prefix="", suffix="", out_float_dtype=np.float64, custom_file=None):
    # save weights to file

    if custom_file is not None:
        out_file = custom_file
    else:
        out_file = ""
        if len(prefix):
            out_file += prefix + "_"
        out_file += "r" + str(rank) + "_j" + str(jack)
        if len(suffix):
            out_file += "_" + suffix
        out_file += ".prm"
        out_file = os.path.join(path, out_file)

    with open(os.path.expanduser(out_file), "w") as f:
        x.astype(out_float_dtype).tofile(f)


def load_weights(rank, jack=1, path=os.getcwd(), prefix="", suffix="", in_float_dtype=np.float64, 
                 out_float_dtype=np.float64, custom_file=None):
    # load weights from file

    if custom_file is not None:
        in_file = custom_file
    else:
        in_file = ""
        if len(prefix):
            in_file += prefix + "_"
        in_file += "r" + str(rank) + "_j" + str(jack)
        if len(suffix):
            in_file += "_" + suffix
        in_file += ".prm"
        in_file = os.path.join(path, in_file)
        
    if os.path.isfile(os.path.expanduser(in_file)):
        with open(os.path.expanduser(in_file), "r") as f:
            x = np.fromfile(f, dtype=in_float_dtype)
    else:
        x = None
            
    return x


def load_checkpoint(rank, jack=1, path=os.getcwd(), prefix="", suffix="", in_float_dtype=np.float64, 
                    out_float_dtype=np.float64, custom_files=None, custom_sampling_history_file=None):
    # load checkpointed data

    chkpnt_file = ""
    if len(prefix):
        chkpnt_file += prefix + "_"
    chkpnt_file += "r" + str(rank) + "_j" + str(jack)
    if len(suffix):
        chkpnt_file += "_" + suffix
    chkpnt_file += ".chk"
    chkpnt_file = os.path.join(path, chkpnt_file)
    
    chkpnt = []
    x = None
    X = None
    E = None
    Fval = None
    if os.path.isfile(os.path.expanduser(chkpnt_file)):
        with open(os.path.expanduser(chkpnt_file), "r") as f:
            for line in f:
                chkpnt.append(line.strip().split(','))
        if isinstance(chkpnt[-1], list):
            chkpnt = chkpnt[-1]
        chkpnt = [int(x) for x in chkpnt]
        if len(chkpnt) > 0:
            x = load_weights(rank, jack, path, prefix, suffix, in_float_dtype, out_float_dtype, custom_files)
            try:
                X, E, Fval = load_sampling_history(rank, jack, path, prefix, suffix, out_float_dtype, 
                                                   custom_sampling_history_file)
            except IOError:
                X = None
                Fval = None

    return x, chkpnt, X, E, Fval



def save_checkpoint(x, rank, iter_num, X=None, E=None, Fval=None, jack=1, path=os.getcwd(), prefix="", suffix="", 
                    out_float_dtype=np.float64, custom_file=None, custom_sampling_history_file=None):
    # save checkpoint

    chkpnt_file = ""
    if len(prefix):
        chkpnt_file += prefix + "_"
    chkpnt_file += "r" + str(rank) + "_j" + str(jack)
    if len(suffix):
        chkpnt += "_" + suffix
    chkpnt_file += ".chk"
    chkpnt_file = os.path.join(path, chkpnt_file)

    chkpnt = [str(iter_num)]

    save_weights(x, rank, jack, path, prefix, suffix, out_float_dtype, custom_file)
    if X is not None:
        save_sampling_history(X, E, Fval, rank, jack, path, prefix, suffix, out_float_dtype, 
                              custom_sampling_history_file)
    
    with open(os.path.expanduser(chkpnt_file), "w") as f:
        f.write(",".join(chkpnt))
    
    return chkpnt



def clear_weights(rank, jack=1, path=os.getcwd(), prefix="", suffix="", custom_file=None):
    # delete weight file

    if custom_file is not None:
        del_file = custom_file
    else:
        if len(prefix):
            del_file += prefix + "_"
        del_file += "r" + str(rank) + "_j" + str(jack)
        if len(suffix):
            del_file += "_" + suffix
        del_file += ".prm"
        del_file = os.path.join(path, del_file)
        
    try:
        os.remove(os.path.expanduser(del_file))
        success = True
    except OSError:
        success = False

    return success


def save_summary(rank, jack=1, ftrain=None, fcv=None, ftest=None, path=os.getcwd(), prefix="", suffix="", 
                 out_float_dtype=np.float64, custom_file=None):
    # save performance numbers to file

    if custom_file is not None:
        out_file = custom_file
    else:
        out_file = ""
        if len(prefix):
            out_file += prefix + "_"
        out_file += "r" + str(rank) + "_j" + str(jack)
        if len(suffix):
            out_file += "_" + suffix
        out_file += ".lst"
        out_file = os.path.join(path, out_file)
        
    fvals = np.ones((3,), dtype=out_float_dtype)*np.nan
    if ftrain is not None:
        fvals[0] = ftrain
    if fcv is not None:
        fvals[1] = fcv
    if ftest is not None:
        fvals[2] = ftest

    with open(os.path.expanduser(out_file), "w") as f:
        fvals.astype(out_float_dtype).tofile(f)


def clear_summary(rank, jack=1, path=os.getcwd(), prefix="", suffix="", custom_file=None):
    # delete summary file

    if custom_file is not None:
        del_file = custom_file
    else:
        del_file = ""
        if len(prefix):
            del_file += prefix + "_"
        del_file += "r" + str(rank) + "_j" + str(jack)
        if len(suffix):
            del_file += "_" + suffix
        del_file += ".lst"
        del_file = os.path.join(path, del_file)
        
    try:
        os.remove(os.path.expanduser(del_file))
        success = True
    except OSError:
        success = False

    return success

        
def load_summary(rank, jack=1, path=os.getcwd(), prefix="", suffix="", in_float_dtype=np.float64, 
                 out_float_dtype=np.float64, custom_file=None):
    # load performance numbers from file (train, cross-validation, test)

    if custom_file is not None:
        in_file = custom_file
    else:
        in_file = ""
        if len(prefix):
            in_file += prefix + "_"
        in_file += "r" + str(rank) + "_j" + str(jack)
        if len(suffix):
            in_file += "_" + suffix
        in_file += ".lst"
        in_file = os.path.join(path, in_file)

    if os.path.isfile(os.path.expanduser(in_file)):
        with open(os.path.expanduser(in_file), "r") as f:
            fvals = np.fromfile(f, dtype=in_float_dtype).astype(out_float_dtype)
    else:
        fvals = [None, None, None]
            
    return fvals[0], fvals[1], fvals[2]


# I/O for Bayesian optimization
def save_sampling_history(X, E, Fval, rank, jack=1, path=os.getcwd(), prefix="", suffix="", out_float_dtype=np.float64, 
                          custom_file=None):
    # save sampling history to file

    if custom_file is not None:
        out_file = custom_file
    else:
        out_file = ""
        if len(prefix):
            out_file += prefix + "_"
        out_file += "hist_r" + str(rank) + "_j" + str(jack)
        if len(suffix):
            out_file += "_" + suffix
        out_file += ".h5"
        out_file = os.path.join(path, out_file)

    h5f = h5py.File(os.path.expanduser(out_file), 'w')
    h5f.create_dataset("X", data=X.astype(out_float_dtype))
    h5f.create_dataset("E", data=E.astype(out_float_dtype))
    h5f.create_dataset("Fval", data=Fval.astype(out_float_dtype))
    h5f.close()


def load_sampling_history(rank, jack=1, path=os.getcwd(), prefix="", suffix="", out_float_dtype=np.float64, 
                          custom_file=None):
    # load sampling history from file

    if custom_file is not None:
        in_file = custom_file
    else:
        in_file = ""
        if len(prefix):
            in_file += prefix + "_"
        in_file += "hist_r" + str(rank) + "_j" + str(jack)
        if len(suffix):
            in_file += "_" + suffix
        in_file += ".h5"
        in_file = os.path.join(path, in_file)

    h5f = h5py.File(os.path.expanduser(in_file), "r")
    X = h5f["X"][:].astype(out_float_dtype)
    E = h5f["E"][:].astype(out_float_dtype)
    Fval = h5f["Fval"][:].astype(out_float_dtype)
    h5f.close()

    return X, E, Fval


def clear_sampling_history(rank, jack=1, path=os.getcwd(), prefix="", suffix="", custom_file=None):
    # clear sampling history file

    if custom_file is not None:
        del_file = custom_file
    else:
        del_file = ""
        if len(prefix):
            del_file += prefix + "_"
        del_file += "hist_r" + str(rank) + "_j" + str(jack)
        if len(suffix):
            del_file += "_" + suffix
        del_file += ".h5"
        del_file = os.path.join(path, del_file)

    try:
        os.remove(os.path.expanduser(del_file))
        success = True
    except OSError:
        success = False

    return success


# weight rescaling functions
def rescale_mned(a, h, J, s_avg, s_std):
    D = s_avg.size

    ap = a + np.sum(s_avg.reshape((D, 1)) * h.reshape((D, 1))) + np.sum(s_avg.reshape((D, 1)) * \
        np.dot(J, s_avg.reshape((D, 1))))
    hp = (h + 2 * np.dot(J, s_avg.reshape((D, 1)))) * s_std.reshape((D, 1))
    Jp = J*np.dot(s_std.reshape((D, 1)), s_std.reshape((1, D)))

    return ap, hp, Jp

def descale_mned(ap, hp, Jp, s_avg, s_std):
    D = s_avg.size

    x = s_avg.reshape((D, 1)) / s_std.reshape((D, 1))
    a = ap - np.sum(hp.reshape((D, 1))*x) + np.sum(x * np.dot(Jp, x))
    h = (hp - 2 * np.dot(Jp, x)) / s_std.reshape((D, 1))
    J = Jp / np.dot(s_std.reshape((D, 1)), s_std.reshape((1, D)))

    return a, h, J

def rescale_mner(a, h, U, V, s_avg, s_std):
    D = s_avg.size
    J = np.dot(U, V.T)

    ap = a + np.sum(s_avg.reshape((D, 1)) * h.reshape((D, 1))) + np.sum(s_avg.reshape((D, 1)) * \
        np.dot(J, s_avg.reshape((D, 1))))
    hp = (h + 2 * np.dot(J, s_avg.reshape((D, 1)))) * s_std.reshape((D, 1))
    Up = U*np.tile(s_std.reshape((D, 1)), (1, U.shape[1]))
    Vp = V*np.tile(s_std.reshape((D, 1)), (1, V.shape[1]))

    return ap, hp, Up, Vp

def descale_mner(ap, hp, Up, Vp, s_avg, s_std):
    D = s_avg.size
    Jp = np.dot(Up, Vp.T)

    x = s_avg.reshape((D, 1)) / s_std.reshape((D, 1))
    a = ap - np.sum(hp.reshape((D, 1)) * x) + np.sum(x * np.dot(Jp, x))
    h = (hp - 2*np.dot(Jp, x)) / s_std.reshape((D, 1))
    U = Up / np.tile(s_std.reshape((D, 1)), (1, Up.shape[1]))
    V = Vp / np.tile(s_std.reshape((D, 1)), (1, Vp.shape[1]))

    return a, h, U, V

# not used
def neg_log_likelihood(s, y, a, h, U, V, epsilon, EPS):
    P = 1.0/(1.0 + np.exp(-a - s.dot(h) - np.sum(s * (s.dot(U.dot(V.T))), axis=1)))

    return -np.mean(y * np.log(P + EPS) + (1 - y) * np.log(1 - P + EPS)) + \
                    epsilon * (np.sum(U[:,-1] ** 2) + np.sum(V[:,-1] ** 2))

def convert_epsilon_to_label(eps, prec=6):

    label = str('%1.6f' % eps)
    dplace = len(label)
    offset = -1
    for i in range(len(label)):
        if label[i] == ".":
            dplace = i
        elif label[i] != "0" and offset == -1:
            offset = i
    while len(label) > dplace:
        if label[-1] == "0":
            label = label[:-1]
        else:
            break
    if offset == -1:
        label = "0d0"
    elif offset < dplace and offset + prec > dplace and offset + prec <= len(label):
        if offset+prec+1 < len(label):
            label = label[:offset + prec + 1]
            label = label[offset:offset + prec + 1] + "d" + str(dplace - offset - prec)
            index = label.index('.')
            label = label[:index] + label[index + 1:]
        else:
            label = label[offset:] + "d" + str(dplace-offset-len(lda1)+1)
            index = label.index('.')
            label = label[:index] + label[index+1:]
    else:
        if offset+prec < len(label):
            label = label[:offset + prec]
            if offset > dplace:
                label = label[offset:offset + prec] + "d" + str(dplace - offset - prec + 1)
            else:
                label = label[offset:offset + prec] + "d" + str(dplace - offset - prec)
        else:
            if offset > dplace:
                label = label[offset:] + "d" + str(dplace - len(label) + 1)
            else:
                label = label[offset:] + "d" + str(dplace - len(label))
    return label
