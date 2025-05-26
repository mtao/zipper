# catch2__dep = dependency('catch2')
catch2_with_main_dep = dependency('catch2-with-main')

test_hello_world = executable('test_hello_world', 'test_hello_world.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('hello world', test_hello_world)

test_iterate = executable('test_iterate', 'test_iterate.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('iterate', test_iterate)
test_matrix = executable('test_matrix', 'test_matrix.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('matrix', test_matrix)
test_tensor = executable('test_tensor', 'test_tensor.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('tensor', test_tensor)
test_extents = executable('test_extents', 'test_extents.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('extents', test_extents)

test_swizzle = executable('test_swizzle', 'test_swizzle.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('swizzle', test_swizzle)

test_slicing = executable('test_slicing', 'test_slicing.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('slicing', test_slicing)

test_span = executable('test_span', 'test_span.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('span', test_span)

test_declare_operations = executable('test_declare_operations', 'test_declare_operations.cpp', dependencies: [zipper_dep,catch2_with_main_dep])

 subdir('storage')
