This is the GitHub Repository for the paper:
["From a single sequence to evolutionary trajectories: protein language models capture the evolutionary potential of SARS-CoV-2 protein sequences"](https://www.biorxiv.org/content/10.1101/2024.07.05.602129v2).
## Table of Contents
1. [Paper Abstract](#abstract)
2. [Installation](#installation)
3. [Figures](#figures)

![Graphical Paper Abstract](https://www.biorxiv.org/content/biorxiv/early/2024/09/18/2024.07.05.602129/F1.large.jpg?width=800&height=600&carousel=1)
---
## Paper Abstract
Protein language models (PLMs) capture features of protein three-dimensional structure from amino acid sequences alone, without requiring multiple sequence alignments (MSA). The concepts of grammar and semantics from natural language have the potential to capture functional properties of proteins. Here, we investigate how these representations enable assessment of variation due to mutation. Applied to SARS-CoV-2’s spike protein using in silico deep mutational scanning (DMS) we demonstrate the PLM, ESM-2, has learned the sequence context within which variation occurs, capturing evolutionary constraint. This recapitulates what conventionally requires MSA data to predict. Unlike other state-of-the-art methods which require protein structures or multiple sequences for training, we show what can be accomplished using an unmodified pretrained PLM. We demonstrate that the grammaticality and semantic scores represent novel metrics. Applied to SARS-CoV-2 variants across the pandemic we show that ESM-2 representations encode the evolutionary history between variants, as well as the distinct nature of variants of concern upon their emergence, associated with shifts in receptor binding and antigenicity. PLM likelihoods can also identify epistatic interactions among sites in the protein. Our results here affirm that PLMs are broadly useful for variant-effect prediction, including unobserved changes, and can be applied to understand novel viral pathogens with the potential to be applied to any protein sequence, pathogen or otherwise.


## Installation
All of the necessary python packages can be found in the requirements.txt file. 

Python version 3.12.0 was used.

In order to use ESM-2 with the Evolocity package, this elif statement has to be added to the model loader section of the 'featurize_seqs.py' file.
```python
elif model_name == 'esm2-small':
    from ..tools.fb_model import FBModel
    model = FBModel(
        'esm2_t33_650M_UR50D',
        repr_layer=[-1],
    )
```
## Figures
Many of the figures in the paper can be found in an interactive form on [Observable](https://observablehq.com/@cvr-bioinfo/from-a-single-sequence-to-evolutionary-trajectorie).