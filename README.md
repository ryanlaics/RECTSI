
# ReCTSi: Resource-efficient Correlated Time Series Imputation via Decoupled Pattern Learning and Completeness-aware Attentions

Welcome to the official repository for the paper "ReCTSi: Resource-efficient Correlated Time Series Imputation via Decoupled Pattern Learning and Completeness-aware Attentions." This repository contains the necessary code, datasets, and supplemental material related to the paper.

## Supplemental Material

For additional insights into our work, including detailed time and space complexity analysis, implementation details, and the ablation study on other datasets, please refer to the [Supplemental Material](Appendix.pdf). We recommend downloading the document for enhanced readability.

## Code and Datasets

### Setting Up the Environment

To set up the required experimental environment, we recommend using Anaconda. Create and activate an environment with the following commands:

```bash
# Create the environment
conda env create -f conda_env.yml

# Activate the environment
conda activate rectsi
```

### Datasets

We have implemented ReCTSi using several datasets, including traffic datasets (PeMS-BA, PeMS-LA, and PeMS-SD), an air quality dataset (AQ36), and an infection case dataset (COVID-19).

Download the datasets from [Google Drive](https://drive.google.com/file/d/1kmY2MMlga1ryasGsAHXslKNI3F2l19IT/) (courtesy of [PoGeVon](https://github.com/Derek-Wds/PoGeVon/) from KDD 2023). After downloading, unzip and move them into the `dataset` folder at the root of this repository.

### Deep Learning-based Baselines

Below is a table of DL-based baseline models for comparison:

| Model    | Venue   | Year | Link                                                  |
|----------|---------|------|-------------------------------------------------------|
| BRITS    | NeurIPS | 2018 | [Link](https://dl.acm.org/doi/10.5555/3327757.3327783)               |
| rGAIN    | AAAI    | 2021 | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/17086)        |
| SAITS    | ESWA    | 2022 | [Link](https://www.sciencedirect.com/science/article/pii/S0957417423001203)    |
| TimesNet | ICLR    | 2022 | [Link](https://openreview.net/pdf?id=ju_Uqw384Oq)    |
| GRIN     | ICLR    | 2022 | [Link](https://openreview.net/pdf?id=kOu3-S3wJ7)       |
| NET<sup>3</sup>   | WWW     | 2021 | [Link](https://dl.acm.org/doi/10.1145/3442381.3449969) |
| PoGeVon  | KDD     | 2023 | [Link](https://dl.acm.org/doi/10.1145/3580305.3599444)            |

### Running Experiments

To run experiments and compute metrics for deep imputation methods, use the `run_imputation.py` script. Here`s an example command:

```bash
python run_imputation.py --config config/rectsi/air36.yaml
```

For experiments with the PEMS datasets, adjust the `subdataset_name` value in the `pems.yaml` configuration file to match the specific dataset (`PEMS-04`,`PEMS-07`,`PEMS-11`).

## Acknowledgements

This project builds upon the work of [GRIN](https://github.com/Graph-Machine-Learning-Group/grin) and [PoGeVon](https://github.com/Derek-Wds/PoGeVon/). We extend our gratitude to their contributions to the field.
