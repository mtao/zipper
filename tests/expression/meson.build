subdir('nullary')
subdir('unary')

test_slicing = executable('test_slicing', 'test_slicing.cpp', dependencies: [zipper_test_dep])
test('slicing', test_slicing)

test_swizzle = executable('test_swizzle', 'test_swizzle.cpp', dependencies: [zipper_test_dep])
test('swizzle', test_swizzle)

test_vector_expr = executable('test_vector_expr', 'test_vector.cpp', dependencies: [zipper_test_dep])
test('vector_expr', test_vector_expr)

test_products = executable('test_products', 'test_products.cpp', dependencies: [zipper_test_dep])
test('products', test_products)

test_homogeneous = executable('test_homogeneous', 'test_homogeneous.cpp', dependencies: [zipper_test_dep])
test('homogeneous', test_homogeneous)
#
subdir('reductions')
