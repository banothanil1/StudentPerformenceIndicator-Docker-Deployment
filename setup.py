from setuptools import find_packages,setup

def get_packages(requirementTxtFile:str)->list[str]:
    '''this function will return the list of packages in list str format based on file given'''
    requirments = []
    with open(requirementTxtFile) as reqObj:
        requirments = [req.replace("\n","") for req in reqObj if req not in ('-e .')]
    return requirments
         
setup(
    name='MLPROJECT-1',
    version='0.0.1',
    author='banoth anil nayak',
    author_email='banothanilnayak50@gmail.com',
    packages= find_packages(),
    install_requires=get_packages('requirements.txt')
)