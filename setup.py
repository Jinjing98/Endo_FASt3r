from setuptools import setup, find_packages

# we lately uninstalled baselines from the cu12 env

# we install reloc3r_uni with editable mode with:
# first go the the dir where setup.py located.
# pip install -e .

setup(
  name = 'reloc3r_uni',  # it will look up /vit_cls with its current hierachy. notice, there shouldn’t be ‘-’ for the pkg name('_'is accepted tho)!!
  packages = find_packages(),  # it will look up the avaible directory where there are __init__.py under it!
  version = '0.0.1',
  license='MIT', 
)
