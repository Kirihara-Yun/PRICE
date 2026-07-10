"""Helpers for deriving convolution input/output shapes and padding.

These routines mirror the semantics of a typical CNN backend: a ``"same"``
padding makes the output spatial size equal to the input size (for stride 1),
while ``"valid"`` means no padding at all.
"""


def get_conv1d_out_shape_and_pad(length, kernel, pad, stride=1, dilation=1):
    if pad == "s":
        pad2 = kernel + (kernel - 1) * (dilation - 1) - 1
        assert stride == 1 and pad2 % 2 == 0, (stride, dilation, kernel)
        return length, pad2 // 2
    return (length + 2 * pad - kernel - (kernel - 1) * (dilation - 1)) // stride + 1, pad


def get_conv1d_out_shape(length, kernel, pad, stride=1, dilation=1):
    return get_conv1d_out_shape_and_pad(length, kernel, pad, stride, dilation)[0]


def get_conv2d_out_shape_and_pad(
    in_shape, kernel_shape, pad_mode="same", stride=(1, 1), dilation=(1, 1)
):
    if pad_mode == "valid":
        pad_mode = (0, 0)
    elif pad_mode == "same":
        pad_mode = ("s", "s")
    else:
        assert not isinstance(pad_mode, str), pad_mode
    rows = (
        get_conv1d_out_shape_and_pad(*pair)
        for pair in zip(in_shape, kernel_shape, pad_mode, stride, dilation)
    )
    return tuple(zip(*rows))


def get_conv2d_out_shape(
    in_shape, kernel_shape, pad_mode="same", stride=(1, 1), dilation=(1, 1)
):
    return get_conv2d_out_shape_and_pad(
        in_shape, kernel_shape, pad_mode, stride, dilation
    )[0]


def to_nchw(shape, layout):
    if layout == "nchw":
        n, c, h, w = shape
    elif layout == "nhwc":
        n, h, w, c = shape
    else:
        assert False
    return n, c, h, w


def from_nchw(shape, layout):
    n, c, h, w = shape
    if layout == "nchw":
        return n, c, h, w
    elif layout == "nhwc":
        return n, h, w, c
    else:
        assert False


def get_conv1d_inp_slice(out_win: slice, out_size, in_size, kernel, pad, stride=1, dilation=1):
    o_start, o_stop, _ = out_size.indices(out_win)
    o_range = o_stop - o_start
    i_range = stride * (o_range - 1) + kernel + (kernel - 1) * (dilation - 1)
    i_start = o_start
    i_stop = i_start + i_range
    i_start = clip(i_start - pad, 0, in_size)
    i_stop = clip(i_stop - pad, 0, in_size)
    return slice(i_start, i_stop)


def clip(value, low, high):
    return min(high, max(low, value))
