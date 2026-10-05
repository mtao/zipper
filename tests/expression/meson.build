subdir('binary')
subdir('nullary')
subdir('unary')

test_dynamic_extents = executable('test_dynamic_extents', 'test_dynamic_extents.cpp', dependencies: [zipper_test_dep])
test('dynamic_extents', test_dynamic_extents)

subdir('reductions')

test_expressions = executable('test_expressions', 'expressions.cpp', dependencies: [zipper_test_dep])
test('expressions', test_expressions)

test_ownership = executable('test_ownership', 'ownership.cpp', dependencies: [zipper_test_dep])
test('ownership', test_ownership)

test_index_set = executable('test_index_set', 'test_index_set.cpp', dependencies: [zipper_test_dep])
test('index_set', test_index_set)

test_zero_aware_operation = executable('test_zero_aware_operation', 'test_zero_aware_operation.cpp', dependencies: [zipper_test_dep])
test('zero_aware_operation', test_zero_aware_operation)

test_assignment_safety = executable('test_assignment_safety', 'assignment_safety.cpp', dependencies: [zipper_test_dep])
test('assignment_safety', test_assignment_safety)

test_tiled_relayout = executable('test_tiled_relayout', 'tiled_relayout.cpp', dependencies: [zipper_test_dep])
test('tiled_relayout', test_tiled_relayout)
test_tiled_relayout_always = executable('test_tiled_relayout_always', 'tiled_relayout.cpp',
  dependencies: [zipper_test_dep],
  cpp_args: ['-DZIPPER_RELAYOUT_TILING_MIN_BYTES=0'])
test('tiled_relayout_always', test_tiled_relayout_always)
