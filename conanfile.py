from conan import ConanFile
from conan.tools.meson import MesonToolchain
from conan.tools.gnu import PkgConfigDeps
import os

__BASE_DEPS__ = [
        "spdlog/1.14.1", 
        "range-v3/cci.20240905",  
        "catch2/3.7.1",
        "mdspan/0.6.0"
        ]





class MultiscreenManga(ConanFile):
    requires = __BASE_DEPS__ 
    settings = "os", "compiler", "build_type"  
    generators = "PkgConfigDeps"



                                               
    def generate(self):
        meson = MesonToolchain(self)                   
        meson.generate()

                                               
