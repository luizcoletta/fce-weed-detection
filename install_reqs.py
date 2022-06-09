import subprocess
import sys

### alternatively you can use the list 'requirements.txt' in the command line: 'pip install -r /path/to/requirements.txt'

### https://pip.pypa.io/en/latest/user_guide/#using-pip-from-your-program
def install_required_packages(modules):
    for module in modules:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', module])


def main():
    external_file = True
    
    if external_file:
        with open('requirements.txt') as f:
            install_requires = f.read().strip().split('\n')
    else:
        install_requires = ['numpy',
                     'scipy',
                     'pandas',
                     'matplotlib',
                     'seaborn',
                     'plotly',
                     'sklearn',
                     'opencv-python',   # cv2
                     'Pillow',          # PIL
                     'tensorflow',
                     'keras',
                     'six',
                     'argparse',
                     'scikit-build',    # skbuild
                     'cmake',
                     'rembg',           # https://github.com/danielgatis/rembg
                     'imgaug',
                     'git+https://github.com/luizfsc/ext-semantic-segmentation']  # keras_segmentation (https://github.com/divamgupta/image-segmentation-keras)
    install_required_packages(install_requires)


if __name__ == "__main__":
    main()
