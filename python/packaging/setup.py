from setuptools import setup

setup(
        # Bare bones
        name="my_library",
        version="0.0.1",
        packages=["my_library"],
        # Metadata that you probably want before publishing, and that
        # setuptools will complain about if you don't have
        author="Me!",
        author_email="Me@example.com",
        url="example.com",
        # Dependencies
        install_requires=["numpy", "scipy>=1.2"],
)
