project(
  'zipper',
  'cpp',
  version: '0.1',
  default_options: ['warning_level=3', 'cpp_std=c++26'],
)

required_deps = []

cpp_compiler = meson.get_compiler('cpp')

compile_args = []

# std::format with ranges support is required (C++26, __cpp_lib_format_ranges)
fmt_code = '''#include <format>
std::string func(int a, int b) { return std::format("{}",std::tuple(a,b)); }
'''
fmt_compiles = cpp_compiler.compiles(fmt_code, name: 'std::format(ranges)')
if not fmt_compiles
  error('Compiler must support std::format with ranges (C++26 __cpp_lib_format_ranges)')
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

# Optional: Doxygen documentation target.
# Build with: ninja -C build-debug docs
doxygen = find_program('doxygen', required: false)
if doxygen.found()
  run_target(
    'docs',
    command: [doxygen, meson.project_source_root() / 'Doxyfile'],
  )
endif
