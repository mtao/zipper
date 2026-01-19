catch2_with_main_dep = dependency('catch2-with-main')

zipper_test_dep = declare_dependency(
  dependencies: [zipper_dep, catch2_with_main_dep],
  compile_args: ['-Wno-padded', '-Wno-float-equal'],
)

subdir('expression')
subdir('storage')
#subdir('utils')
#subdir('detail')
#
## hello_world refurses to not show a strict overflow message without putting it in the cpp args :(. TODO: figure this out.
## hello_world.cpp has extra pragma pushes just to show this issue
#test_hello_world = executable(
#  'test_hello_world',
#  ['test_hello_world.cpp', 'test_struct.cpp'],
#  dependencies: [zipper_test_dep],
#  cpp_args: ['-Wno-strict-overflow'],
#)
#test('hello world', test_hello_world)
#
#base_class_sources = ['test_vector.cpp', 'test_matrix.cpp', 'test_tensor.cpp', 'test_array.cpp']
#test_base_classes = executable('test_base_classes', base_class_sources, dependencies: [zipper_test_dep])
#test('base_classes', test_base_classes)
#
#test_extents = executable('test_extents', 'test_extents.cpp', dependencies: [zipper_test_dep])
#test('extents', test_extents)
#
#test_declare_operations = executable(
#  'test_declare_operations',
#  ['test_expressions.cpp', 'test_declare_operations.cpp'],
#  dependencies: [zipper_test_dep],
#)
