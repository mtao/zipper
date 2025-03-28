project('zipper', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++20'])




fmt_dep = dependency('fmt')

spdlog_dep = dependency('spdlog')

mdspan_dep = dependency('mdspan')

range_v3_dep = dependency('range-v3')

required_deps = [spdlog_dep, range_v3_dep, fmt_dep, mdspan_dep]

sources = []

include_dirs = [include_directories('include')]
zipper_lib = library('zipper', sources, include_directories: include_dirs, dependencies: required_deps)

zipper_dep = declare_dependency(link_with: zipper_lib, dependencies: required_deps, include_directories: include_dirs)


subdir('tests')
subdir('examples')
