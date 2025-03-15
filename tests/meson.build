# catch2__dep = dependency('catch2')
catch2_with_main_dep = dependency('catch2-with-main')

test_hello_world = executable('test_hello_world', 'test_hello_world.cpp', dependencies: [uvl_dep,catch2_with_main_dep])
test('hello world', test_hello_world)

test_iterate = executable('test_iterate', 'test_iterate.cpp', dependencies: [uvl_dep,catch2_with_main_dep])
test('iterate', test_iterate)
test_matrix = executable('test_matrix', 'test_matrix.cpp', dependencies: [uvl_dep,catch2_with_main_dep])
test('matrix', test_matrix)
test_extents = executable('test_extents', 'test_extents.cpp', dependencies: [uvl_dep,catch2_with_main_dep])
test('extents', test_extents)

test_declare_operations = executable('test_declare_operations', 'test_declare_operations.cpp', dependencies: [uvl_dep,catch2_with_main_dep])
