subdir('binary')
subdir('nullary')
subdir('unary')
subdir('reductions')

test_expressions = executable('test_expressions', 'expressions.cpp', dependencies: [zipper_test_dep])
test('expressions', test_expressions)

test_ownership = executable('test_ownership', 'ownership.cpp', dependencies: [zipper_test_dep])
test('ownership', test_ownership)
