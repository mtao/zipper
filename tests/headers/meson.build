umbrella_header_sources = [
  'concepts_all.cpp',
  'transform_all.cpp',
  'unary_math_all.cpp',
  'solver_error.cpp',
  'solver_result.cpp',
  'shape_validation.cpp',
]

foreach source : umbrella_header_sources
  target_name = 'header_smoke_' + source.underscorify().replace('_cpp', '')
  smoke_test = executable(target_name, source, dependencies: [zipper_dep])
  test(target_name, smoke_test)
endforeach
