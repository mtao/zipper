sources = [
  'cross_product.cpp',
  'matrix_product.cpp',
  'addition.cpp',
  'coeff_product.cpp',
  'tensor_product.cpp',
  'form_tensor_product.cpp',
]
test_binary_views = executable(
  'test_binary_views',
  sources,
  dependencies: [zipper_test_dep],
  cpp_args: ['-Wno-strict-overflow'],
)
test('binary_views', test_binary_views)
