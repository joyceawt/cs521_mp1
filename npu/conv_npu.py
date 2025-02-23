import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
# Using stanford reference,. INPUT_DEPTH = input_channels
# LAYER_NUM_FILTERS = output_channels


@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    # gemm_moving_fmax is max number of pixels that can be processed in one cycle
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array/ allocate space to be stored in HBM
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    # num tiles = total number of channels / tile size (max number of channels - 128)
    c_in_pmax = nl.tile_size.pmax  # 128; max number of channels
    c_out_pmax = c_in_pmax  # 128
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    # TODO:
    # 1. Reshape the weights to the required shape
    # 2. Compute the convolution output (use nl.matmul to do the multiply-accumulate, may need multiple calls)
    # 3. Store the result in X_out

    # For SBUF: split data into tiles so each chunk can be loaded into SBUF from HBM using nl.load
    # Load each tile, compute, then store partila result back to HBM using nl.store
    # For PSUM: partial sums can be accumulated in PSUM. Then copy ifnal sums back to SBUF or HBM when done

    ##############################

    # 1) Reshape weights and break out_channels and in_channels into multiple tiles (6D shape)
    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in,
                  c_in_pmax, filter_height, filter_width))

    # 2) Allocate space for weights in SBUF (6D shape)
    # nl.par_dim -> partition dimension of c_out_pmax(128)
    w_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in,
                        c_in_pmax, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)

    # 3) Loop and load weights from HBM to SBUF
    for out_c_tile in nl.affine_range(n_tiles_c_out):
        w_sbuf[out_c_tile] = nl.load(W[out_c_tile])

    assert (in_channels * filter_height *
            filter_width) <= 128, "You need chunking!"

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        raise RuntimeError("Please fill your implementation of computing convolution"
                           " of X[b] with the weights W and bias b and store the result in X_out[b]")

    return X_out
