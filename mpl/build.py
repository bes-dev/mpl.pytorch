import os

from torch.utils.ffi import create_extension

sources = ['src/lib_mpl.cpp']
headers = ['src/lib_mpl.h']
with_cuda = False

this_file = os.path.dirname(os.path.realpath(__file__))

ffi = create_extension(
    '_mpl',
    headers=headers,
    sources=sources,
    relative_to=__file__,
    with_cuda=with_cuda
)

if __name__ == '__main__':
    ffi.build()
