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
        'bin/ti-avg-uncertainty',
        'bin/ti-bs',
        'bin/ti-extract',
        'bin/ti-ga',
        'bin/ti-ga-test',
        'bin/ti-ga-uncertainty',
        'bin/ti-ga-multi',
        'bin/ti-ga-multi-unc',
        'bin/ti-ga-multi-unc-per-day',
        'bin/ti-ga-multi-test',
        'bin/ti-mc',
        'bin/ti-mc-test',
        'bin/ti-unsga3',
        'bin/ti-unsga3-test',
    ],
    zip_safe=False
)
