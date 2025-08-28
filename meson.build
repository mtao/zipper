project('zipper', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++26'])

required_deps = []




cpp_compiler = meson.get_compiler('cpp')


compile_args = []


if not cpp_compiler.has_define('__cpp_lib_format_ranges')
  message('current version of std::format is insufficient, fetching dependency')
  fmt_dep = dependency('fmt')
  required_deps += fmt_dep
endif

# Waiting for fmt to work properly on compilers. don't trust the macro
fmt_code = '''#include <format>
std::string func(int a, int b) { std::format("{}",std::tuple(a,b); }
'''

fmt_compiles = cpp_compiler.compiles(fmt_code, name : 'std::format(ranges)')
if not fmt_compiles
  compile_args += '-DZIPPER_FMT_OVERRIDES_DISABLED'
endif

if not cpp_compiler.has_define('__cpp_lib_mdspan')
  message('current version of mdspan is insufficient, fetching dependency')
  if get_option('use_conan')
    mdspan_dep = dependency('mdspan')
  else
    mdspan_proj = subproject('mdspan')
    mdspan_dep = mdspan_proj.get_variable('mdspan_dep')
  endif
  required_deps += mdspan_dep
endif


if get_option('use_conan')
  range_v3_dep = dependency('range-v3')
else
  range_v3_proj = subproject('range-v3')
  range_v3_dep = range_v3_proj.get_variable('range_v3_dep')
  required_deps += range_v3_dep
endif


sources = []

include_dirs = [include_directories('include')]

zipper_dep = declare_dependency(dependencies: required_deps, include_directories: include_dirs, compile_args: compile_args)


if get_option('testing')
  subdir('tests')
endif
if get_option('examples')
  subdir('examples')
endif
