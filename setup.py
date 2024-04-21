from setuptools import setup

def read_requirements():
    with open('requirements.txt', 'r') as req:
        return req.read().splitlines()

setup(
    name="neural-network",
    version="1.0",
    description="Python implementation of an artificial neural network aimed at deeply mastering the underlying concepts that drive neural networks. This project is inspired by Andrej Karpathy and Russ Salakhutdinov.",
    author="Malo Le Mestre",
    author_email="malo.lm@icloud.com",
    packages=["neural_network"],
    install_requires=read_requirements(),
)