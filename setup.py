import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hybrid_recommender",
    version = '0.4',
    author="Devendra Kumar Sahu",
    author_email="devsahu99@gmail.com",
    description="This package will create recommendations based on content as well as user ratings and finally providing top recommendations based on both data points",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devsahu99/hybrid_recommender",
    packages=['hybrid_recommender'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
