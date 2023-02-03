from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='ti',
    version='0.1',
    description='Tremor inversion',
    long_description=readme(),
    url='http://github.com/lvanderlaat/ti',
    author='Leonardo van der Laat',
    author_email='laat@umich.edu',
    packages=['ti'],
    install_requires=[
    ],
    scripts=[
        'bin/ti-avg',
        'bin/ti-bs',
        'bin/ti-extract',
        'bin/ti-ga',
        'bin/ti-mc',
        'bin/ti-post',
        'bin/ti-unsga3',
    ],
    zip_safe=False
)
