subdir('reductions')
test_views = executable('test_views', ['test_vector.cpp', 'test_products.cpp', 'test_homogeneous.cpp'], dependencies: [zipper_dep,catch2_with_main_dep])
test('views', test_views)
test_swizzle = executable('test_swizzle', 'test_swizzle.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('swizzle', test_swizzle)

test_slicing = executable('test_slicing', 'test_slicing.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('slicing', test_slicing)

test_span = executable('test_span', 'test_span.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('span', test_span)
