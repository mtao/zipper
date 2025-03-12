# uvl
unnamed vector library


```bash
conan install . --output-folder=build/conan --build=missing # prepare conan
pushd build # enter build directory
meson setup --native-file conan/conan_meson_native.ini .. . # configure meson to use the output of conan
ninja # build
```

