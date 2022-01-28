from setuptools import setup

setup(name='visualizer',
	version='1.0.0',
	description='A package for visualizing SemNet results',
	author='Stephen Allegri',
	author_email='sallegri3@gatech.edu',
	packages=['visualizer'],
	install_requires=[
        'dash==2.0.0', 
        'dash_cytoscape==0.2.0', 
        'dash_html_components==1.0.1', 
        'dash_core_components==1.3.1', 
        'dash_table==4.4.1', 
        'dash_bootstrap_components==1.0.1', 
        'numpy==1.21.2', 
        'pandas==1.3.4', 
        'networkx==2.6.3', 
        'colour==0.1.5']
)