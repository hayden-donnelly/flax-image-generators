from setuptools import setup, find_packages

setup(
    name = 'flax-image-generators',
    packages=['flax_image_generators'],
    version = '0.0.0',
    license='Apache-2.0',
    description = 'Image generation models implemented in Flax.',
    long_description_content_type = 'text/markdown',
    author = 'Hayden Donnelly',
    author_email = 'donnellyhd@outlook.com',
    url = 'https://github.com/hayden-donnelly/flax-image-generators',
    install_requires=[
        'flax>=0.7.2',
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
    ],
)