sources = [
  'determinant.cpp',
  'trace.cpp',
  'coefficient_reductions.cpp',
  'norms.cpp',
]
test_view_reductions = executable('test_view_reductions', sources, dependencies: [zipper_test_dep])
test('view_reductions', test_view_reductions)

test_contraction = executable('test_contraction', 'contraction.cpp', dependencies: [zipper_test_dep])
test('contraction', test_contraction)
