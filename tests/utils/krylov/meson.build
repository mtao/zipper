krylov_sources = [
  'power_method.cpp',
  'arnoldi.cpp',
  'lanczos.cpp',
]
test_krylov = executable('test_krylov', krylov_sources, dependencies: [zipper_test_dep])
test('krylov', test_krylov)
