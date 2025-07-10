catch2_with_main_dep = dependency('catch2-with-main')

sources = ['partial_reduction.cpp']
test_unary_views = executable('test_unary_views', sources, dependencies: [zipper_dep,catch2_with_main_dep])
test('unary_views', test_unary_views)
