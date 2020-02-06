# Welcome
Oh hey.

# Description

This is the code for EE5940, Optimal Control and Reinforcement Learning.

# Stuff to install

* [git](https://git-scm.com/) - For managing code. If you use a Mac or Linux, you probably have it.

* [anaconda 3.7](https://www.anaconda.com/download/) - For code. There are two versions on the webpage, 2.7 and 3.7. This class uses 3.6. (If you already have a working Python 3 installation, you won't need this. However, you will need `jupyter`, `numpy`, `scipy`, `matplotlib`)

* A few Python libraries. Anaconda has most of what we need. The rest can be installed in the command line 

```
pip3 install pyglet
pip3 install cvxpy
pip3 install tensorflow
pip3 install casadi
```

# Getting the Course Files

## Initial setup

This command creates a directory with the course files.

```
git clone https://github.umn.edu/ee5940/EE5940.git
```

## Getting the new files

When the files update, you can get them by using

```
git pull
```

# Running the Homework

## Important: Make renamed copy of the assignent

The homeworks are in Jupyter notebooks. They will have filenames like `homework1.ipynb`.

Before you get started on an assignment you should copy it to something with a different filename. For example:

```
cp homework1.ipynb myHomework1.ipynb
```

The reason for this is as follows. If the homework file needs to change due to a bug, when you try to download the new file, it will attempt to over-write your file. (Andy will try to remember to change the filenames to prevent this, but he might forget.) Renaming your file reduces the likelihood of this issue.

(Andy once wrote a script to automate this renaming process. However, ensuring that this script works on every computer (and in particular, every Windows version) is a nightmare of its own.)



## Running the notebook file

The homeworks are in Jupyter notebooks. You can open the notebook from the command line (while in the class directory)

```
jupyter notebook
```

This will open up a browser tab/window. Click on the filename and follow the instructions. 

# Turning in the Homework

The notebooks will be uploaded to Canvas. Please download the file from your computer's file manager. Do not use "Save Link As" from the browser. This gives the HTML file rendered in the, rather than the notebook file, which contains the code.

