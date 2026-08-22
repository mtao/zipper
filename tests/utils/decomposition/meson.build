decomposition_sources = [
  'qr.cpp',
  'llt.cpp',
  'ldlt.cpp',
  'lu.cpp',
  'svd.cpp',
  'polar.cpp',
]
test_decomposition = executable('test_decomposition', decomposition_sources, dependencies: [zipper_test_dep])
test('decomposition', test_decomposition)
