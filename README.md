<!--
  Copyright (c) 2023 - 2025 TNO-ESI
  All rights reserved.
-->

# MBDlyb: A Python library for Model-Based Diagnostics
This repository contains the prototype library resulting from the SD2Act project on design-driven model-based diagnostics.

## Structure
MBDlyb is structured in several packages:
* **mbdlyb** contains the base classes use throughout modelling formalisms. Additionally, _gdb_ consists the base classes for storing the knowledge graphs created in MBDlyb in a [Neo4j](https://neo4j.com/) graph database.
* **formalisms** contains different reasoning formalisms supported by the library. These classes are wrappers around the reasoning formalisms used to compute diagnoses.
* **ui** contains the base classes for creating a web interface to allow users to create and use their diagnostic models.
* **functional** contains the function-oriented modeling paradigm for diagnostics. This paradigm supports three reasoning formalisms (Bayesnet, Markov random fields and tensor networks) and the diagnostic models can be stored in a graph database. Also, a web interface has been implemented in the _mbdlyb.functional.ui_ package. More paradigms may be added in the future.

## Example
An example model in the form of a simple CDRadioPlayer can be found in _tests/test_model.py_.

## Packaging and installing
To build MBDlyb, the [Hatch](https://hatch.pypa.io/) project manager is required.
This can be installed by running:
`pip install hatch`

MBDlyb can be packaged into a wheel package with the following command:  
`hatch build -t wheel`

The wheel package can be installed using the following command:  
`pip install <path_to_whl>`

### Versioning
Upon releasing a small feature update or bugfix, one can update the micro-version:
`hatch version micro`

For a larger update, one may use either `hatch version minor` or `hatch version major`.

## Usage
The interactive model editor and diagnoser can be started with the following command:
`python -m mbdlyb`

Next open a webbrower and go to [http://localhost:8080/]. This will open the model editor main page in which you can either open an existing model, create a new model or import an Excel/Json or Capella model. 

From the model editor page of a top-level cluster the interactive diagnoser and the design for diagnostics page can be opened by clicking the respective icons.

## Contact information
For more information or questions about the library and the methodology supported by it, please contact [Thomas Nägele](mailto:thomas.nagele@tno.nl).