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

# ── Optional: SuiteSparse (CHOLMOD, UMFPACK, SPQR) ─────────────────────
suitesparse_opt = get_option('use_suitesparse')
suitesparse_deps = []
have_suitesparse = false

if suitesparse_opt.disabled()
  message('SuiteSparse support disabled by user')
else
  # Try pkg-config / cmake first, then fall back to find_library + has_header.
  cholmod_dep  = dependency('CHOLMOD',  required: false)
  umfpack_dep  = dependency('UMFPACK',  required: false)
  spqr_dep     = dependency('SPQR',     required: false)

  if cholmod_dep.found() and umfpack_dep.found() and spqr_dep.found()
    suitesparse_deps = [cholmod_dep, umfpack_dep, spqr_dep]
    have_suitesparse = true
  else
    # Manual fallback: check for headers + libraries
    _ss_libs = []
    _ss_ok = true
    foreach _lib : ['cholmod', 'umfpack', 'spqr', 'suitesparseconfig']
      _dep = cpp_compiler.find_library(_lib, required: false)
      if _dep.found()
        _ss_libs += _dep
      else
        _ss_ok = false
      endif
    endforeach

    if _ss_ok and cpp_compiler.has_header('cholmod.h') and cpp_compiler.has_header('umfpack.h') and cpp_compiler.has_header('SuiteSparseQR.hpp')
      suitesparse_deps = _ss_libs
      have_suitesparse = true
    endif
  endif

  if have_suitesparse
    message('SuiteSparse found — enabling CHOLMOD/UMFPACK/SPQR bindings')
    compile_args += '-DZIPPER_HAS_SUITESPARSE'
    required_deps += suitesparse_deps
  elif suitesparse_opt.enabled()
    error('SuiteSparse requested but not found')
  else
    message('SuiteSparse not found — sparse direct solvers disabled')
  endif
endif

zipper_dep = declare_dependency(
  dependencies: required_deps,
  include_directories: include_dirs,
  compile_args: compile_args,
)
meson.override_dependency('zipper', zipper_dep)

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
