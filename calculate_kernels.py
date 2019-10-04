"""
Compute deep kernels with a bunch of different hyperparameters and save them to
disk
"""
import numpy as np
import tensorflow as tf
from gpflow import settings
import deep_ckern as dkern
import tqdm
import sys
import os
import gpflow


def create_kern(ps):
    if ps['seed'] == 1234:
        return dkern.DeepKernel(
            [1, 28, 28],
            filter_sizes=[[5, 5], [2, 2], [5, 5], [2, 2]],
            recurse_kern=dkern.ExReLU(multiply_by_sqrt2=True),
            var_weight=1.,
            var_bias=1.,
            padding=["VALID", "SAME", "VALID", "SAME"],
            strides=[[1, 1]] * 4,
            data_format="NCHW",
            skip_freq=-1,
        )

    if 'skip_freq' not in ps:
        ps['skip_freq'] = -1
    if ps['nlin'] == 'ExReLU':
        recurse_kern = dkern.ExReLU(multiply_by_sqrt2=True)
    else:
        recurse_kern = dkern.ExErf()
    return dkern.DeepKernel(
        [1, 28, 28],
        filter_sizes=[[ps['filter_sizes'], ps['filter_sizes']]] * ps['n_layers'],
        recurse_kern=recurse_kern,
        var_weight=ps['var_weight'],
        var_bias=ps['var_bias'],
        padding=ps['padding'],
        strides=[[ps['strides'], ps['strides']]] * ps['n_layers'],
        data_format="NCHW",
        skip_freq=ps['skip_freq'],
    )


def resize_to_fit(out, results):
    #print("out's shape: {}    results's shape: {}".format(out.shape, results.shape))
    shape = (len(results[0]),) + out
    print("required shape:", shape)
    out = np.zeros(shape, dtype=np.float32)
    return out


def compute_big_Kdiag(out, sess, kern, n_max, X, n_gpus=1):
    N = X.shape[0]
    slices = list(slice(j, j+n_max) for j in range(0, N, n_max))
    K_ops = []
    for i in range(n_gpus):
        with tf.device("gpu:{}".format(i)):
            X_ph = tf.placeholder(settings.float_type, [None, X.shape[1]], "X_ph")
            Kdiag = kern.Kdiag(X_ph)
            if not isinstance(Kdiag, list):
                # Ensure output is a batch of vectors
                Kdiag = tf.expand_dims(Kdiag, 0)
            K_ops.append((X_ph, Kdiag))

    out_to_create_or_resize = True
    for j in tqdm.trange(0, len(slices), n_gpus):
        feed_dict = {}
        ops = []
        for (X_ph, Kdiag), j_s in zip(K_ops, slices[j:j+n_gpus]):
            feed_dict[X_ph] = X[j_s]
            ops.append(Kdiag)
        results = sess.run(ops, feed_dict=feed_dict)

        if out_to_create_or_resize:
            out_to_create_or_resize = False
            out = resize_to_fit(out, results)

        for r, j_s in zip(results, slices[j:j+n_gpus]):
            out[:, j_s] = r
    return out


def compute_big_K(out, sess, kern, n_max, X, X2=None, n_gpus=1):
    N = X.shape[0]
    N2 = N if X2 is None else X2.shape[0]

    # Make a list of all the point kernel matrices to be computed
    if X2 is None or X2 is X:
        diag_symm = True
        slices = list((slice(j, j+n_max), slice(i, i+n_max))
                      for j in range(0, N, n_max)
                      for i in range(j, N2, n_max))
    else:
        diag_symm = False
        slices = list((slice(j, j+n_max), slice(i, i+n_max))
                      for j in range(0, N, n_max)
                      for i in range(0, N2, n_max))

    # Make the required kernel ops and placeholders for each GPU
    K_ops = []
    for i in range(n_gpus):
        with tf.device("gpu:{}".format(i)):
            X_ph = tf.placeholder(settings.float_type, [None, X.shape[1]], "X_ph")
            X2_ph = tf.placeholder(settings.float_type, X_ph.shape, "X2_ph")
            K_cross = kern.K(X_ph, X2_ph)
            if not isinstance(K_cross, list):
                # Ensure output is a batch of matrices
                K_cross = tf.expand_dims(K_cross, 0)

            if diag_symm:
                K_symm = kern.K(X_ph, None)
                if not isinstance(K_symm, list):
                    # Ensure output is a batch of matrices
                    K_symm = tf.expand_dims(K_symm, 0)
            else:
                K_symm = None

            K_ops.append((X_ph, X2_ph, K_cross, K_symm))

    out_to_create_or_resize = True
    # Execute on all GPUs concurrently
    for j in tqdm.trange(0, len(slices), n_gpus):
        feed_dict = {}
        ops = []
        for (X_ph, X2_ph, K_cross, K_symm), (j_s, i_s) in (
                zip(K_ops, slices[j:j+n_gpus])):
            if j_s == i_s and diag_symm:
                feed_dict[X_ph] = X[j_s]
                ops.append(K_symm)
            else:
                feed_dict[X_ph] = X[j_s]
                if X2 is None:
                    feed_dict[X2_ph] = X[i_s]
                else:
                    feed_dict[X2_ph] = X2[i_s]
                ops.append(K_cross)
        results = sess.run(ops, feed_dict=feed_dict)

        if out_to_create_or_resize:
            out_to_create_or_resize = False
            out = resize_to_fit(out, results)
            print("New shape:", out.shape)

        for r, (j_s, i_s) in zip(results, slices[j:j+n_gpus]):
            out[:, j_s, i_s] = r
    return out

def create_array_dataset(diag, N, N2):
    """
    Returns a tuple which has the required dimensions for the array holding the
    kernel matrix.
    """
    return (N,) if diag else (N, N2)


def calculate_K(name, X, X2, diag, kern, n_gpus=1, n_max=400, chunk_size_MB=32):
    sess = gpflow.get_default_session()
    
    N = X.shape[0]
    if X2 is None:
        N2 = N
    else:
        N2 = X2.shape[0]
    
    print("Computing {}".format(name))
    out = create_array_dataset(diag, N, N2)
    if diag:
        out = compute_big_Kdiag(out, sess, kern, n_max=n_max, X=X, n_gpus=n_gpus)
    else:
        out = compute_big_K(out, sess, kern, n_max=n_max, X=X, X2=X2, n_gpus=n_gpus)

    return out