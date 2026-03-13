sources = [
  #
  'partial_reduction.cpp',
  'repeat.cpp',
  'cast.cpp',
  'scalar_arithmetic.cpp',
  'reshape.cpp',
  'cofactor.cpp',
]
test_unary_views = executable('test_unary_views', sources, dependencies: [zipper_test_dep])
test('unary_views', test_unary_views)

test_anti_slice = executable('test_anti_slice', 'anti_slice.cpp', dependencies: [zipper_test_dep])
test('anti_slice', test_anti_slice)
