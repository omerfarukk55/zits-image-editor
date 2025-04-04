from setuptools import setup, find_packages

setup(
    name="goruntu-isleme-app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask>=2.0.1",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "torch",
        "torchvision",
        "lama-cleaner",
        "werkzeug>=2.0.0",
    ],
    python_requires=">=3.7",
) 