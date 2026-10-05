test_format = executable('test_format', 'format.cpp', dependencies: [zipper_test_dep])
test('format', test_format)

test_determinant = executable('test_determinant', 'determinant.cpp', dependencies: [zipper_test_dep])
test('determinant', test_determinant)

test_minmax = executable('test_minmax', 'minmax.cpp', dependencies: [zipper_test_dep])
test('minmax', test_minmax)

test_inverse = executable('test_inverse', 'inverse.cpp', dependencies: [zipper_test_dep])
test('inverse', test_inverse)

# std::execution::par in libstdc++ needs TBB; the parallel test is compiled
# only when it can be linked.
tbb_dep = dependency('tbb', required: false)
test_gemm_kernel = executable('test_gemm_kernel', 'test_gemm_kernel.cpp',
  dependencies: [zipper_test_dep, tbb_dep],
  cpp_args: tbb_dep.found() ? ['-DZIPPER_TEST_HAVE_TBB'] : [])
test('gemm_kernel', test_gemm_kernel)

test_gemm_tuning = executable('test_gemm_tuning', 'test_gemm_tuning.cpp',
  dependencies: [zipper_test_dep])
test('gemm_tuning', test_gemm_tuning)

test_matrix_utils = executable('test_matrix_utils', 'test_matrix_utils.cpp', dependencies: [zipper_test_dep])
test('matrix_utils', test_matrix_utils)

subdir('extents')
subdir('decomposition')
subdir('solver')
subdir('krylov')
subdir('eigenvalue')
subdir('orthogonalization')
subdir('sparse')
