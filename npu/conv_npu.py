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

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0
    assert out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    # gemm_moving_fmax is max number of pixels that can be processed in one cycle
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array/ allocate space to be stored in HBM
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    # num tiles = total number of channels / tile size (max number of channels - 128)
    c_in_pmax = nl.tile_size.pmax  # 128; max number of channels
    c_out_pmax = c_in_pmax  # 128
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax

    # 1) Reshape weights and break out_channels and in_channels into multiple tiles (6D shape)
    W = W.reshape((n_tiles_c_out, c_out_pmax, n_tiles_c_in,
                  c_in_pmax, filter_height, filter_width))

    # 2) Allocate space for weights in SBUF (6D shape)
    # nl.par_dim -> partition dimension of c_out_pmax(128)
    w_sbuf = nl.ndarray((n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in,
                        c_in_pmax, filter_height, filter_width), dtype=W.dtype, buffer=nl.sbuf)

    # 3) Loop and load weights from HBM to SBUF
    for oc_tile in nl.affine_range(n_tiles_c_out):
        w_sbuf[oc_tile] = nl.load(W[oc_tile])

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        x_sbuf = nl.ndarray((n_tiles_c_in, nl.par_dim(c_in_pmax), input_height, input_width),
                            dtype=X.dtype, buffer=nl.sbuf)

        # Load input tiles into SBUF
        for ic_tile in nl.affine_range(n_tiles_c_in):
            start_c = ic_tile * c_in_pmax
            end_c = start_c + c_in_pmax
            x_sbuf[ic_tile] = nl.load(X[b, start_c:end_c, :, :])

        # Process each output channel tile
        for oc_tile in nl.affine_range(n_tiles_c_out):
            psum = nl.zeros((nl.par_dim(c_out_pmax), out_height,
                            out_width), dtype=X.dtype, buffer=nl.psum)

            # Process input channel tiles
            for ic_tile in nl.affine_range(n_tiles_c_in):
                # Process each filter tile
                for fh in nl.affine_range(filter_height):
                    for fw in nl.affine_range(filter_width):
                        # Load the weights for this tile
                        w_tile = w_sbuf[oc_tile, :, ic_tile, :, fh, fw]

                        # Extract input window
                        window = x_sbuf[ic_tile, :, fh:fh +
                                        out_height, fw:fw+out_width]

                        psum += nl.matmul(w_tile, window)

            # 4) Add bias and store the result in HBM
            bias_slice = bias[oc_tile * c_out_pmax: (oc_tile + 1) * c_out_pmax]
            result = nisa.tensor_scalar(psum, nl.add, bias_slice)
            X_out[b, oc_tile * c_out_pmax: (oc_tile + 1)
                  * c_out_pmax] = nl.store(result)

    return X_out
