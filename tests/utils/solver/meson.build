solver_sources = [
  'iterative.cpp',
  'triangular_solve.cpp',
  'pcg.cpp',
  'multigrid.cpp',
]
test_solver = executable('test_solver', solver_sources, dependencies: [zipper_test_dep])
test('solver', test_solver)
