from setuptools import setup, find_packages

setup(
    name="llm-news-classifier",
    version="0.1",
    packages=find_packages(),
    license="MIT",
    author="Matin Khajavi",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)