from setuptools import setup, find_packages

__package_name__ = "eclare"


def get_version_and_cmdclass(pkg_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(pkg_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.get_cmdclass(pkg_path)


__version__, cmdclass = get_version_and_cmdclass(__package_name__)


# noinspection PyTypeChecker
setup(
    name=__package_name__,
    version=__version__,
    description="ECLARE: Efficient cross-planar learning for anisotropic resolution enhancement",
    long_description="ECLARE: Efficient cross-planar learning for anisotropic resolution enhancement",
    author="Samuel W. Remedios",
    author_email="samuel.remedios@jhu.edu",
    url="https://github.com/sremedios/ECLARE",
    license="Apache License, 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    packages=find_packages(),
    keywords="mri ct image super-resolution",
    entry_points={
        "console_scripts": [
            "run-eclare=eclare.main:main",
        ]
    },
    install_requires=[
        "nibabel",
        "numpy",
        "scipy",
        "torch>=2.1",
        "tqdm",
        "radifox-utils==1.0.3",
        "matplotlib",
    ],
    cmdclass=cmdclass,
)
