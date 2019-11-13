from setuptools import find_packages, setup

_REQUIREMENTS = ['numpy', 'Pillow', 'matplotlib', 'scikit-image', 'imageio', 
                'opencv-python', 'h5py', 'imgaug', 'pickle-mixin',
                'tensorflow', 'keras', 'pandas', 'pillow']

setup(
    name='trainer',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=_REQUIREMENTS,
)