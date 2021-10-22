from setuptools import setup

setup(name='cdc',
      version='0.1',
      description='cdc_quant_research',
      url='https://github.com/crowang/cdcqr',
      author='Wang Han',
      author_email='wang.han@crypto.com',
      packages=['cdcqr'],
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
