project('zipper', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++26'])




fmt_dep = dependency('fmt')

spdlog_dep = dependency('spdlog')

if option('use_conan')
  mdspan_dep = dependency('mdspan')
else
  mdspan_proj = subproject('mdspan')
  mdspan_dep = mdspan_proj.get_variable('mdspan_dep')
endif

range_v3_dep = dependency('range-v3')

required_deps = [spdlog_dep, range_v3_dep, fmt_dep, mdspan_dep]

sources = []

include_dirs = [include_directories('include')]

zipper_dep = declare_dependency(dependencies: required_deps, include_directories: include_dirs)


if get_option('testing')
  subdir('tests')
endif
if get_option('examples')
  subdir('examples')
endif
