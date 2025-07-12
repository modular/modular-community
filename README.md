# Modular Community Channel ✨

Welcome to the repository for the Modular community channel! This conda-based [Prefix.dev](http://Prefix.dev) channel allows Modular community members to distribute their packages built with MAX and Mojo via Pixi 🪄

## Installing a package

### Add the Modular community channel to your `pixi.toml` file

Before you can install a community package, you’ll need to add the Modular community channel to your `pixi.toml` file.

Add the Modular community channel (https://repo.prefix.dev/modular-community) to your `pixi.toml file` in the channels section:

```
# pixi.toml

[project]
channels = ["conda-forge", "https://conda.modular.com/max", "https://repo.prefix.dev/modular-community"]
description = "Add a short description here"
name = "my-mojo-project"
platforms = ["osx-arm64"]
version = "0.1.0"

[tasks]

[dependencies]
max = ">=25.4.0"
```

### **Install a package**

To install a package from the Modular community channel, simply enter the following in the command line:
```
pixi add <name of the package.
```

That’s it! Your package is installed. To double-check that the correct package has been installed, run:
```
pixi list
```
This command will list all packages installed in your project.

## Submitting a package

To submit your package to the Modular community channel, you’ll need to:
1. Fork the Modular community channel GitHub repository
2. Add a folder to the `/recipes` folder. Give it the same name as your package.
3. In the folder for your package, include a rattler-build recipe file named `recipe.yaml` and a file that includes tests for your package.
