import setuptools

with open("README.md", "r", encoding="utf-8") as f:
  long_description = f.read()

setuptools.setup(
  name="mvnet",
  version="0.0.1",
  author="borgwang",
  author_email="badbobobo@gamil.com",
  description="A lightweight deep learning library",
  long_description="A small but fully functional deep learning framework",
  long_description_content_type="text/markdown",
  url="https://github.com/borgwang/mvnet",
  packages=setuptools.find_packages(),
  classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
  ],
  install_requires=["numpy", "networkx"],
  python_requires=">=3.8",
  extras_require={
    "gpu": ["pyopencl"],
    "cuda": ["pycuda"],
    "linting": ["flake8", "pylint", "mypy", "pre-commit"],
    "testing": ["pytest"],
  }
)
