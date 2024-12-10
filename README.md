# Modular Community Channel ✨

Welcome to the repository for the Modular community channel! This conda-based [Prefix.dev](http://Prefix.dev) channel allows Modular community members to distribute their packages built with MAX and Mojo via Magic 🪄

We’re currently running an Early Access Program, and we’d love for you to join the fun! Whether you want to install packages, contribute your own, or review creations from fellow community members, it’s easy to get started. Simply [join our forum](https://forum.modular.com/t/community-channel-early-access/213) and [opt in](https://forum.modular.com/t/community-channel-early-access/213) – you'll receive a DM shortly with more info.

## Installing a package

### Add the Modular community channel to your `mojoproject.toml` file

Before you can install a community package, you’ll need to add the Modular community channel to your `mojoproject.toml` or `pixi.toml` file.

Add the Modular community channel (https://repo.prefix.dev/modular-community) to your `mojoproject.toml` file or `pixi.toml file` in the channels section:

```
# mojoproject.toml or pixi.toml

[project]
channels = ["conda-forge", "https://conda.modular.com/max", "https://repo.prefix.dev/modular-community"]
description = "Add a short description here"
name = "my-mojo-project"
platforms = ["osx-arm64"]
version = "0.1.0"

[tasks]

[dependencies]
max = ">=24.5.0,<25"
```

### **Install a package**

To install a package from the Modular community channel, simply enter the following in the command line:
```
magic add <name of the package.
```

That’s it! Your package is installed. To double-check that the correct package has been installed, run:
```
magic list
```
This command will list all packages installed in your project.

## Submitting a package

To submit your package to the Modular community channel, you’ll need to:
1. Fork the Modular community channel GitHub repository
2. Add a folder to the `/recipes` folder. Give it the same name as your package.
3. In the folder for your package, include a rattler-build recipe file named `recipe.yaml` and a file that includes tests for your package.

Before submitting a package, please [join the early access program](https://forum.modular.com/t/community-channel-early-access/213) to get access to the full instructions and guidelines (plus, you'll get swag!).
