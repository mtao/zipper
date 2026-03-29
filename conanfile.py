from conan import ConanFile
from conan.tools.meson import MesonToolchain
from conan.tools.gnu import PkgConfigDeps

class Zipper(ConanFile):
    requires = [
        "fmt/11.2.0",
        "range-v3/0.12.0",
        "catch2/3.8.1",
        "mdspan/0.6.0",
    ]
    settings = "os", "compiler", "build_type", "arch"
    generators = "PkgConfigDeps"

    def generate(self):
        meson = MesonToolchain(self)
        meson.generate()
