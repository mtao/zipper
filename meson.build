project(
  'zipper',
  'cpp',
  version: '0.1',
  default_options: ['warning_level=3', 'cpp_std=c++26'],
)

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

fmt_compiles = cpp_compiler.compiles(fmt_code, name: 'std::format(ranges)')
if not fmt_compiles
  compile_args += '-DZIPPER_FMT_OVERRIDES_DISABLED'
endif

alias_ctad_code = '''template <typename A> struct B {B(A) {}};
template <typename A> B(A) -> B<A>;
template <typename A> using C = B<A>;
void f() {
  C c(int{});
}
'''

alias_ctad_compiles = cpp_compiler.compiles(alias_ctad_code, name: 'aliased_CTAD')
if not alias_ctad_compiles
  compile_args += '-DZIPPER_DONT_USE_ALIAS_CTAD'
endif

if not cpp_compiler.has_define('__cpp_lib_mdspan')
  message('current version of mdspan is insufficient, fetching dependency')
  if get_option('use_mdspan_subproject')
    mdspan_proj = subproject('mdspan')
    mdspan_dep = mdspan_proj.get_variable('mdspan_dep')
  else
    mdspan_dep = dependency('mdspan')
  endif
  required_deps += mdspan_dep
endif

#range_v3_dep = dependency('range-v3', required: false)
#if not range_v3_dep.found()
#  if get_option('use_ranges_subproject')
#    ranges_v3_proj = subproject('ranges-v3')
#    ranges_v3_dep = ranges-v3_proj.get_variable('ranges-v3_dep')
#  else
#    ranges_v3_dep = dependency('ranges-v3')
#  endif
#endif
#required_deps += range_v3_dep

sources = []

include_dirs = [include_directories('include')]

zipper_dep = declare_dependency(
  dependencies: required_deps,
  include_directories: include_dirs,
  compile_args: compile_args,
)

if get_option('testing')
  subdir('tests')
endif
#if get_option('examples')
#  subdir('examples')
#endif
