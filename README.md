
# Zipper

A zippy C++26 tensor processing library. Heavily inspired from [Eigen](https://eigen.tuxfamily.org/) and [mdpsan](https://cppreference.com/utility/mdspan), this library-in-development intends to provide a light-weight clean syntax for linear algebra and tensor algebra while also providing convenient backends to transform between different data representations.

## Principles

The underlying concept in Zipper is a _view_, which merely represents an underlying map $$\mathbb{Z}^r \rightarrow T$$ for some rank $$r$$. When a view represents a (multi-dimensional) array of objects. The semantics of vectors (r=1), matrices (r=2), tensors, or forms are induced by wrapping that view in a container for which those semantics are induced.

More concretely, we can take unit vectors as

```cpp
#include <zipper/views/nullary/UnitView.hpp>

// vector {0,1,0}
auto e1 = zipper::views::nullary::unit_view<double,3>(1);
```

This view can be used by multiple types of semantics, like the semantics of a vector or form. Note that below deduction guides are used to maintain proper ownership of the view (e1 is a non-const ref so the `*Base` classes do not own any data, but instead hold a reference to `e1` internally).

```cpp
#include <zipper/VectorBase.hpp>
#include <zipper/FormBase.hpp>

// a vector whose data is the view e1
zipper::VectorBase x = e1;

// a 1-form, equivalently a row-vector whose data is the view e1
zipper::FormBase u = e1;
```

With these semantic wrappers we can now do semantic-specific operations

```cpp
// form * vector is like a dot product, resulting in a scalar
double res = u * x;

// a vector holding a view that holds the expression for (2 * x)
zipper::VectorBase x2 = 2 * x;

// a vector that owns its own data
zipper::Vector x2_ = x2;
```

### Expressions and Value Categories

Those who are used to Eigen will recognize this structure of building expression templates using CRTP. The main difference here is that the semantics kept separate from the inheritance heirarchy and instead class membership is used. This has the subtle advantage of letting us be careful about value categories - deduction guides can now let us store r-value arguments by value rather than by reference, making complex arguments safer.
See

```cpp
zipper::Vector<double,4> x = {0,1,2,3};
auto p = x + 3 * zipper::Vector<double,4>({2,3,4,5});
```

results in `p` being stored as something like

```
struct Addition {
    zipper::Vector<double,4>& lhs; // = x
    struct ScalarProduct {
        double lhs;// = 3
        zipper::Vector<double,4> rhs; // = {2,3,4,5}
    } rhs;
};
```

, where every value declared in the same line as `p`'s declaration is stored by value rather than by reference.
In Eigen, from my experience, the internal ScalarProduct would be stored as a reference, so the temporary object would disappear at the end of the line and the expression would therefore point to invalid memory.

## Dependencies

Zipper depends on [fmt](https://fmt.dev) and [mdspan](https:://github.com/kokkos/mdspan), but both of these dependencies should disappear as c++26 functionality becomes more common in existing libraries.

For buliding and testing zipper currently depends on [meson](https://mesonbuild.com)
and [Catch2](https://catch2.org), and is ready for use with [conan](https://conan.io).

## Testing

If you have your own installations of all of the above dependencies you can
build by

```bash
meson setup .. . -Dtesting=true
meson test -C build # build and test
```

```bash
# prepare conan
conan install . --output-folder=build/conan --build=missing 
# enter build directory
pushd build 
# configure meson to use the output of conan
meson setup --native-file conan/conan_meson_native.ini .. . -Dtesting=true 
ninja # build
```
