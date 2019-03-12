# Introduction 
Q&A matcher project

# Getting Started
This is a Python project

# Build and Test
Use pipenv and pytest

Locally you can go into ./src directory and use pipenv to install.

NOTE: You'll need to be on VPN to install all the packages.  Some are coming from our own pypi server. 

```
pipenv install 
```

Or you can alternatively use docker-compose.yml to build the containers. 

```
docker-compose up
```

# Upgrading dependencies
Use pipenv to upgrade a package.  Ensure you are on VPN.  Choose a new branch if required and run the upgrade.

```
pipenv upgrade hu_training
```

Now commit the changed pipfile.lock to the repo

# Contribute
TODO: