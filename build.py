import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from typing import Any, Dict


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeExtensionBuilder(build_ext):
    def build_extensions(self) -> None:
        self.prepare_cmake_extensions()
        super().build_extensions()

    def build_extension(self, ext: Extension) -> None:
        if isinstance(ext, CMakeExtension):
            self.build_cmake_extension(ext)
        else:
            super().build_extension(ext)

    def prepare_cmake_extensions(self) -> None:
        cmake_extensions = [x for x in self.extensions if isinstance(x, CMakeExtension)]
        if cmake_extensions:  # pragma: no branch
            try:
                out = subprocess.check_output(["cmake", "--version"])
            except OSError:  # pragma: no cover
                raise RuntimeError(
                    "CMake must be installed to build the following extensions: "
                    + ", ".join(e.name for e in cmake_extensions)
                )

    def build_cmake_extension(self, ext: CMakeExtension) -> None:
        ext_full_path = self.get_ext_fullpath(ext.name)
        dist_version = self.distribution.metadata.get_version()
        extdir = Path(ext_full_path).parent.resolve()
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":  # pragma: no cover
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j4"]
        cmake_args += ["-DPYTHON_INCLUDE_DIR={}".format(sysconfig.get_path("include"))]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), dist_version
        )
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", ext.name.rpartition(".")[-1]]
            + build_args,
            cwd=self.build_temp,
        )


ext_modules = [
    # .suffix must match cmake module
    CMakeExtension(f"leap.LeapAccelerate", sourcedir=".")
]


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        dict(
            ext_modules=ext_modules,
            cmdclass=dict(build_ext=CMakeExtensionBuilder),
            zip_safe=False,
        )
    )
