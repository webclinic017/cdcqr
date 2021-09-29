from setuptools import setup

setup(name='croqr',
      version='0.1',
      description='crypto_quant_research',
      url='https://github.com/crowang/croqr',
      author='Wang Han',
      author_email='wang.han@crypto.com',
      packages=['croqr'],
      install_requires=[
          'pytest==4.4.0',
          'pytest-runner==4.4'
      ],
      setup_requires=[
          'pytest-runner'
      ],
      tests_require=[
          'pytest'
      ],
      zip_safe=False)
