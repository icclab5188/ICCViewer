# coding=utf-8
import glob
import os
import subprocess
import sys

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension

version = "0.1.0"
package_name = "groundingdino"
cwd = os.path.dirname(os.path.abspath(__file__))

sha = "Unknown"
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
except Exception:
    pass

def write_version_file():
    version_path = os.path.join(cwd, "groundingdino", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        # f.write(f"git_version = {repr(sha)}\n")

def get_extensions():
    extensions_dir = os.path.join(cwd, "groundingdino", "models", "GroundingDINO", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)

    sources = list(set([main_source] + sources))
    extension = CppExtension
    define_macros = []
    extra_compile_args = {"cxx": []}

    if CUDA_HOME and torch.cuda.is_available():
        print("Compiling with CUDA")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            '-allow-unsupported-compiler',
            '-gencode=arch=compute_89,code=sm_89'
        ]
    else:
        print("CUDA not available or CUDA_HOME not set. Skipping CUDA extensions.")
        return []

    include_dirs = [extensions_dir]
    sources = list(set(sources))

    ext_modules = [
        extension(
            name="groundingdino._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

def parse_requirements(fname="requirements.txt", with_version=True):
    import re
    import sys
    from os.path import exists

    def parse_line(line):
        if line.startswith("-r "):
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]
                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield from parse_line(line)

    def gen_packages_items():
        if exists(fname):
            for info in parse_require_file(fname):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                yield "".join(parts)

    return list(gen_packages_items())

if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version} on Windows")

    with open("LICENSE", "r", encoding="utf-8") as f:
        license_text = f.read()

    write_version_file()

    setup(
        name=package_name,
        version=version,
        author="International Digital Economy Academy, Shilong Liu",
        url="https://github.com/IDEA-Research/GroundingDINO",
        description="Open-set object detector",
        license=license_text,
        install_requires=parse_requirements("requirements.txt"),
        packages=find_packages(exclude=["configs", "tests"]),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
