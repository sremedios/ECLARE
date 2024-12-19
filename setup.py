from setuptools import setup, find_packages

__package_name__ = "eclare"

# noinspection PyTypeChecker
setup(
    name=__package_name__,
    version=1.0.0,
    description="ECLARE: Efficient cross-planar learning for anisotropic resolution enhancement",
    long_description="ECLARE: Efficient cross-planar learning for anisotropic resolution enhancement",
    author="Samuel W. Remedios",
    author_email="samuel.remedios@jhu.edu",
    url="https://github.com/sremedios/ECLARE",
    license="Apache License, 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
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
    cmdclass={},
)
