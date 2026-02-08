catch2_with_main_dep = dependency('catch2-with-main')

zipper_test_dep = declare_dependency(
  dependencies: [zipper_dep, catch2_with_main_dep],
  compile_args: ['-Wno-padded', '-Wno-float-equal'],
  include_directories: ['.'],

)

subdir('expression')
subdir('storage')
subdir('utils')
subdir('detail')

test_struct = executable('test_struct', 'test_struct.cpp', dependencies: [zipper_test_dep])
test('struct', test_struct)

test_array = executable('test_array', 'test_array.cpp', dependencies: [zipper_test_dep])
test('array', test_array)
#
test_extents = executable('test_extents', 'test_extents.cpp', dependencies: [zipper_test_dep])
test('extents', test_extents)

test_declare_operations = executable(
  'test_declare_operations',
  ['test_declare_operations.cpp'],
  dependencies: [zipper_test_dep],
)
test('declare_operations', test_declare_operations)

test_vector = executable('test_vector', 'test_vector.cpp', dependencies: [zipper_test_dep])
test('vector', test_vector)

test_concepts = executable('test_concepts', 'test_concepts.cpp', dependencies: [zipper_test_dep])
test('concepts', test_concepts)

test_matrix = executable('test_matrix', 'test_matrix.cpp', dependencies: [zipper_test_dep])
test('matrix', test_matrix)

test_tensor = executable('test_tensor', 'test_tensor.cpp', dependencies: [zipper_test_dep])
test('tensor', test_tensor)

test_expressions = executable('test_expressions', 'test_expressions.cpp', dependencies: [zipper_test_dep])
test('expressions', test_expressions)

test_span_construction = executable('test_span_construction', 'span_construction.cpp', dependencies: [zipper_test_dep])
test('span_construction', test_span_construction)

test_zipperbase_stl = executable('test_zipperbase_stl', 'zipperbase_stl.cpp', dependencies: [zipper_test_dep])
test('zipperbase_stl', test_zipperbase_stl)

test_hello_world = executable(
  'test_hello_world',
  ['test_hello_world.cpp'],
  dependencies: [zipper_test_dep],
  cpp_args: ['-Wno-strict-overflow'],
)
test('hello_world', test_hello_world)
