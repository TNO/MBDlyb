<!--
  Copyright (c) 2023 - 2025 TNO-ESI
  All rights reserved.
-->

# MBDlyb
MBDlyb is a prototype Python library resulting for design-driven model-based diagnostics. It allows for experimentation at scale by easily specifying large networks. The library formalizes and consolidates the research done in the SD2Act project ([report](https://repository.tno.nl/SingleDoc?docId=68341)). For more information and some high-level documentation, pleae visit [our website](https://esi.nl/research/output/tools/mbdlyb).

# Installation & usage
This section describes how to install MBDlyb, how to setup the database ([Neo4j](https://neo4j.com/)) which is used to store the models, and how to start MBDlyb.

## Installation
MBDlyb is developed in Python and distributed as a Python wheel package. It is advised to install MBDlyb in a Python virtual environment, e.g., using `venv` or `conda`. How to create a virtual environment using `venv` is described below.

### Create a virtual environment
First create a directory for your MBDlyb installation, and open a shell (e.g., Windows PowerShell) in this directory. This can be done using the following commands:

```bash
mkdir C:\MBDlyb\
cd C:\MBDlyb\
```

Run the following command to create a virtual environment using `venv` named `mbdlyb_env`:

```bash
python -m venv mbdlyb_env
```

### Installing MBDlyb
Open a shell (e.g., Windows PowerShell) in the directory with the virtual environment, and activate the virtual environment:

```bash
cd C:\MBDlyb\
.\mbdlyb_env\Scripts\Activate.ps1
```

The MBDlyb wheel package can be installed using the following command:

```bash
pip install <path_to_whl>
```

This will install MBDlyb, and download and install all its dependencies.

## MBDlyb usage
As a prerequisite, make sure the [Neo4j](https://neo4j.com/) database engine is running.

The interactive model editor and diagnoser can be started from the command line.
Open a shell (e.g., Windows PowerShell) in the directory with the virtual environment created for MBDlyb, and activate it:

```bash
cd C:\MBDlyb\
.\mbdlyb_env\Scripts\Activate.ps1
```

MBDlyb can now be started with the following command. Make sure the database name in the `--database` argument matches the database created in the previous step.

```bash
python -m mbdlyb --database=example-db
```

Note that this command uses the default hostname, port, protocol, username and password. These can also be specified using the arguments `--host`, `--port`, `--protocol`, `--username` and `--password`.

Next open a webbrowser and go to <http://localhost:8080/>. This will open the model editor main page in which you can either open an existing model, create a new model or import an Excel/Json or Capella model.

To get a list of all command line parameters, and instructions on how to use them, run the following command:

```bash
python -m mbdlyb --help
```

### Example
An example model in the form of a simple CDRadioPlayer can be found in _tests/test_model.py_.

## Library structure
MBDlyb is structured in several packages:
* **mbdlyb** contains the base classes use throughout modelling formalisms. Additionally, _gdb_ contains the base classes for storing the knowledge graphs created in MBDlyb in a Neo4j graph database.
* **formalisms** contains different reasoning formalisms supported by the library. These classes are wrappers around the reasoning formalisms used to compute diagnoses.
* **ui** contains the base classes for creating a web interface to allow users to create and use their diagnostic models.
* **functional** contains the function-oriented modeling paradigm for diagnostics. This paradigm supports three reasoning formalisms (Bayesnet, Markov random fields and tensor networks) and the diagnostic models can be stored in a graph database. Also, a web interface has been implemented in the _mbdlyb.functional.ui_ package. More paradigms may be added in the future.

## Packaging
To build MBDlyb, the [Hatch](https://hatch.pypa.io/) project manager is required.
This can be installed by running:
`pip install hatch`

MBDlyb can be packaged into a wheel package with the following command:  
`hatch build -t wheel`

### Versioning
Upon releasing a small feature update or bugfix, one can update the micro-version:
`hatch version micro`

For a larger update, one may use either `hatch version minor` or `hatch version major`.

## Contact information
For more information or questions about the library and the methodology supported by it, please contact [Thomas Nägele](mailto:thomas.nagele@tno.nl).