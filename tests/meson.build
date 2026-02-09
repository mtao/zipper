catch2_with_main_dep = dependency('catch2-with-main')

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
