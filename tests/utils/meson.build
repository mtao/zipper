test_format = executable('test_format', 'format.cpp', dependencies: [zipper_test_dep])
test('format', test_format)

test_determinant = executable('test_determinant', 'determinant.cpp', dependencies: [zipper_test_dep])
test('determinant', test_determinant)

test_minmax = executable('test_minmax', 'minmax.cpp', dependencies: [zipper_test_dep])
test('minmax', test_minmax)

test_inverse = executable('test_inverse', 'inverse.cpp', dependencies: [zipper_test_dep])
test('inverse', test_inverse)

test_matrix_utils = executable('test_matrix_utils', 'test_matrix_utils.cpp', dependencies: [zipper_test_dep])
test('matrix_utils', test_matrix_utils)

subdir('extents')
subdir('decomposition')
subdir('solver')
subdir('krylov')
subdir('eigenvalue')
subdir('sparse')
