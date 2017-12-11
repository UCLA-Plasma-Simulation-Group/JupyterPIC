[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/UCLA-Plasma-Simulation-Group/JupyterPIC/master)

Jupyter notebooks for education plasma physics simulations with PIC.

We welcome contributions and ideas:  please email us at picksc.org@gmail.com

To run this on your local computer:

* Docker must be installed
* Open a terminal and navigate into this directory
* Execute `docker build -t osiris-class -f Dockerfile-local .`
* Execute `./runosdocker`
* This will output several lines of text to your terminal, including a web URL
* Paste the web URL into a web browser
* Navigate to any of the project directories and open the *.ipynb files
* To quit, "Logout" of the Jupyter notebook, close your browser windows, and (if necessary) type Ctrl-C twice in your terminal window.

The above commands will write simulation output into the project directories.  If you would like to keep a fresh copy of this original GitHub repository, it is important that you copy this JupyterPIC directory into a separate location on your computer for simulation purposes and personal file modifications.

