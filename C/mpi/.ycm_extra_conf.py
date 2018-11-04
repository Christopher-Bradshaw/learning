def FlagsForFile(filename, **kwargs):
  return {
    'flags': ['c', '-Wall', '-Wextra', '-Werror', '-I/usr/include/openmpi-x86_64/', '-L/usr/lib64/openmpi/lib/', '-lmpi'],
  }
