def calculate_contiguity(shape, strides):
  # https://github.com/numpy/numpy/blob/93a97649aa0aefc0ee8ee5fc7cb78063bfe67255/numpy/core/src/multiarray/flagsobject.c#L115
  assert len(shape) == len(strides)
  ndim = len(shape)
  c_contiguous = f_contiguous = True
  if ndim:
    nitems = 1
    for i in range(ndim-1, -1, -1):
      if shape[i] == 0:
        return True, True
      if shape[i] != 1:
        if strides[i] != nitems:
          c_contiguous = False
        nitems *= shape[i]
    nitems = 1
    for i in range(ndim):
      if shape[i] != 1:
        if strides[i] != nitems:
          f_contiguous = False
        nitems *= shape[i]
  return c_contiguous, f_contiguous

def calculate_slices(start, stop, step, length):
  # https://github.com/python/cpython/blob/d034590294d4618880375a6db513c30bce3e126b/Objects/sliceobject.c#L264
  assert step != 0, "Slice step cannot be zero"
  if step is None: step = 1
  if start is None: start = length+1 if step < 0 else 0
  if stop is None: stop = -length-1 if step < 0 else length+1

  if start < 0:
    start += length
    if start < 0: start = -1 if step < 0 else 0
  elif start >= length:
    start = length-1 if step < 0 else length
  if stop < 0:
    stop += length
    if stop < 0: stop = -1 if step < 0 else 0
  elif stop >= length:
    stop = length-1 if step < 0 else length

  if step < 0 and stop < start:
    size = (start - stop - 1) // (-step) + 1
  elif step > 0 and start < stop:
    size = (stop - start - 1) // (step) + 1
  else:
    size = 0
  return start, stop, step, size
