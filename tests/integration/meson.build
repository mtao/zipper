test_hello_world = executable(
  'test_hello_world',
  'hello_world.cpp',
  dependencies: [zipper_test_dep],
  cpp_args: ['-Wno-strict-overflow'],
)
test('hello_world', test_hello_world)
