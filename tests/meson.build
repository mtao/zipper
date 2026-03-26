catch2_with_main_dep = dependency('catch2-with-main', default_options: ['tests=false'])

zipper_test_dep = declare_dependency(
  dependencies: [zipper_dep, catch2_with_main_dep],
  compile_args: ['-Wno-padded', '-Wno-float-equal'],
  include_directories: ['.'],

)

subdir('concepts')
subdir('expression')
subdir('integration')
subdir('storage')
subdir('utils')
subdir('detail')
subdir('transform')

test_struct = executable('test_struct', 'struct.cpp', dependencies: [zipper_test_dep])
test('struct', test_struct)

test_array = executable('test_array', 'array.cpp', dependencies: [zipper_test_dep])
test('array', test_array)

test_vector = executable('test_vector', 'vector.cpp', dependencies: [zipper_test_dep])
test('vector', test_vector)

test_matrix = executable('test_matrix', 'matrix.cpp', dependencies: [zipper_test_dep])
test('matrix', test_matrix)

test_span_construction = executable('test_span_construction', 'span_construction.cpp', dependencies: [zipper_test_dep])
test('span_construction', test_span_construction)

test_zipperbase_stl = executable('test_zipperbase_stl', 'zipperbase_stl.cpp', dependencies: [zipper_test_dep])
test('zipperbase_stl', test_zipperbase_stl)

test_data_array = executable('test_data_array', 'data_array.cpp', dependencies: [zipper_test_dep])
test('data_array', test_data_array)

test_function_boundary = executable('test_function_boundary', 'test_function_boundary.cpp', dependencies: [zipper_test_dep])
test('function_boundary', test_function_boundary)

test_readme_examples = executable('test_readme_examples', 'readme_examples.cpp', dependencies: [zipper_test_dep])
test('readme_examples', test_readme_examples)

test_sparse_types = executable('test_sparse_types', 'sparse_types.cpp', dependencies: [zipper_test_dep])
test('sparse_types', test_sparse_types)
