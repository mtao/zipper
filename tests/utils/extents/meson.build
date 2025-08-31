
srcs = ['offset.cpp', 'dynamic.cpp']
test_extents_utils= executable('test_extents_utils', srcs, dependencies: [zipper_test_dep])
test('test_extents_utils', test_extents_utils)
test_iterate = executable('test_iterate', 'test_iterate.cpp', dependencies: [zipper_test_dep])
test('iterate', test_iterate)
