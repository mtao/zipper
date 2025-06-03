# catch2__dep = dependency('catch2')
catch2_with_main_dep = dependency('catch2-with-main')

test_hello_world = executable('test_hello_world', 'test_hello_world.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('hello world', test_hello_world)

test_iterate = executable('test_iterate', 'test_iterate.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('iterate', test_iterate)

test_tensor = executable('test_tensor', ['test_vector.cpp','test_matrix.cpp', 'test_tensor.cpp'], dependencies: [zipper_dep,catch2_with_main_dep])
test('tensors', test_tensor)

test_extents = executable('test_extents', 'test_extents.cpp', dependencies: [zipper_dep,catch2_with_main_dep])
test('extents', test_extents)



test_declare_operations = executable('test_declare_operations', 'test_declare_operations.cpp', dependencies: [zipper_dep,catch2_with_main_dep])

 subdir('storage')
 subdir('views')
 subdir('utils')
