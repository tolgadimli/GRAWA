import numpy as np
import sys
sys.path.append('../')
from tqdm import tqdm
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator
from scipy.sparse.linalg import eigsh
from warnings import warn
import torch
import time
import logging


def tridiag_to_eigv(tridiag_list):
  """Preprocess the tridiagonal matrices for density estimation.
  Args:
    tridiag_list: Array of shape [num_draws, order, order] List of the
      tridiagonal matrices computed from running num_draws independent runs
      of lanczos. The output of this function can be fed directly into
      eigv_to_density.
  Returns:
    eig_vals: Array of shape [num_draws, order]. The eigenvalues of the
      tridiagonal matricies.
    all_weights: Array of shape [num_draws, order]. The weights associated with
      each eigenvalue. These weights are to be used in the kernel density
      estimate.
  """
  # Calculating the node / weights from Jacobi matrices.
  num_draws = len(tridiag_list)
  num_lanczos = tridiag_list[0].shape[0]
  eig_vals = np.zeros((num_draws, num_lanczos))
  all_weights = np.zeros((num_draws, num_lanczos))
  for i in range(num_draws):
    try:
        nodes, evecs = np.linalg.eigh(tridiag_list[i])
    except np.linalg.LinAlgError:
        return eig_vals, all_weights


    index = np.argsort(nodes)
    nodes = nodes[index]
    evecs = evecs[:, index]
    eig_vals[i, :] = nodes
    all_weights[i, :] = evecs[0] ** 2
  return eig_vals, all_weights

def lanczos(func, dim, max_itr, use_cuda=False, verbose=False):
    r'''
        Lanczos iteration following the wikipedia article here
            https://en.wikipedia.org/wiki/Lanczos_algorithm
        Inputs:
            func   : model functional class
            dim    : dimensions
            max_itr: max iteration
            use_gpu: Use Gpu
            verbose: print extra details
        Outputs:
            eigven values
            weights
    '''
    float_dtype = torch.float64

    # Initializing empty arrays for storing
    tridiag = torch.zeros((max_itr, max_itr), dtype=float_dtype)
    vecs = torch.zeros((dim, max_itr), dtype=float_dtype)

    # intialize a random unit norm vector
    init_vec = torch.zeros((dim), dtype=float_dtype).uniform_(-1, 1)
    init_vec /= torch.norm(init_vec)
    vecs[:, 0] = init_vec

    # placeholders for data
    beta = 0.0
    v_old = torch.zeros((dim), dtype=float_dtype)

    for k in tqdm(range(max_itr)):
        #t = time.time()
        #print(k)
        v = vecs[:, k]
        if use_cuda:
            v = v.type(torch.float32).cuda()
        time_mvp = time.time()
        w = func.hvp(v)
        if use_cuda:
            v = v.cpu().type(float_dtype)
            w = w.cpu().type(float_dtype)
        time_mvp = time.time() - time_mvp

        w -= beta * v_old
        alpha = np.dot(w, v)
        tridiag[k, k] = alpha
        w -= alpha*v

        # Reorthogonalization
        for j in range(k):
            tau = vecs[:, j]
            coeff = np.dot(w, tau)
            w -= coeff * tau

        beta = np.linalg.norm(w)

        if beta < 1e-6:
            break
            quit()

        if k + 1 < max_itr:
            tridiag[k, k+1] = beta
            tridiag[k+1, k] = beta
            vecs[:, k+1] = w / beta

        v_old = v

        # info = f"Iteration {k} / {max_itr} done in {time.time()-t:.2f}s (MVP: {time_mvp:.2f}s)"
        # if (verbose) and (k%10 == 0):
        #     logger = logging.getLogger('my_log')
        #     logger.info(info)

    return vecs, tridiag


def eig_trace(model_func, max_itr, draws, use_cuda=False, verbose=False):
    tri = np.zeros((draws, max_itr, max_itr))
    
    for num_draws in tqdm(range(draws)):
        _, tridiag = lanczos(model_func, model_func.dim, max_itr, use_cuda, verbose)
        tri[num_draws, :, :] = tridiag.numpy()

    e, w = tridiag_to_eigv(tri)
    e = np.mean(e, 0)
    return max(e), sum(e), e[-1]/e[len(e)-5] 