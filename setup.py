import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

pkgs = setuptools.find_packages()
print('found these packages:', pkgs)

pkg_name = "ml_ms4alg_snippets"

setuptools.setup(
    name=pkg_name,
    version="0.1.11",
    author="Geoffrey Barrett",
    author_email="gmbarrett313@gmail.com",
    description="Mountainsort v4 for MountainLab - Snippets Version",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeoffBarrett/ml_ms4alg_snippets",
    packages=pkgs,
    package_data={
        '': ['*.mp'],  # Include all processor files
    },
    install_requires=
    [
        'pybind11',
        'isosplit5',
        'numpy',
        'mountainlab_pytools',
        'h5py'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    conda={
        "build_number":0,
        "build_script":[
            "python -m pip install --no-deps --ignore-installed .",
            "CMD=\"ln -sf $SP_DIR/"+pkg_name+" `CONDA_PREFIX=$PREFIX ml-config package_directory`/"+pkg_name+"\"",
            "echo $CMD",
            "$CMD"
        ],
        "test_commands":[
            "ml-list-processors",
            "ml-spec ms4alg_snippets.sort"
        ],
        "test_imports":[
        ],
        "requirements":[
            "python",
            "pip",
            "pybind11",
            "isosplit5",
            "numpy",
            "mountainlab",
            "mountainlab_pytools",
            "h5py"
        ]
    }
)