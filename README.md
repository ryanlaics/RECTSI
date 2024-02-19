<meta name="robots" content="noindex">

<h1> ReCTSi: Resource-efficient Correlated Time Series Imputation via Decoupled Pattern Learning and Completeness-aware Attentions </h1>

This is the repository of the paper "ReCTSi: Resource-efficient Correlated Time Series Imputation via Decoupled Pattern Learning and Completeness-aware Attentions", encompassing the code, datasets, and supplemental material.

<h1> Supplemental Material </h1> 

Detailed time and space complexity analysis, implementation details, and ablation study on other datasets can be found at the [Appendix](Appendix.pdf) (downloading to local pdf viewer is recommended for better readability).

 <br>
 <br>

  

<h1> Code and Datasets </h1> 

<h2> Requirements </h2> 

To install the experimental environment, please use Anaconda, and create an environment by:
```setup
conda env create -f conda_env.yml
```

Then, activate the environment by:

```activate the environment
conda activate rectsi
```


<h2> Datasets </h2> 

ReCTSi is implemented on three traffic datasets (PeMS-BA, PeMS-LA, and PeMS-SD), an air quality dataset (AQ36), and an infection case dataset (COVID-19).

- **PEMS04**, **PEMS08**, **METR-LA**, and **PEMS-BAY** can be downloaded in [Google Drive](https://drive.google.com/file/d/1kmY2MMlga1ryasGsAHXslKNI3F2l19IT/) (This is provided by [PoGeVon](https://github.com/Derek-Wds/PoGeVon/) (KDD 2023)). After downloading and unzipping the datasets, please move them into the `dataset` folder under the root of this repo.
<h2> DL-based Baselines </h2> 

| Model    | Conference | Year | Link                                                  |
|----------|------------|------|-------------------------------------------------------|
| BRITS    | NeurIPS    | 2018 | https://dl.acm.org/doi/10.5555/3327757.3327783               |
| rGAIN    | AAAI       | 2021 | https://ojs.aaai.org/index.php/AAAI/article/view/17086        |
| SAITS    | ESWA       | 2022 | https://www.sciencedirect.com/science/article/pii/S0957417423001203    |
| TimesNet | ICLR       | 2022 | https://openreview.net/pdf?id=ju_Uqw384Oq    |
| GRIN     | ICLR       | 2022 | https://openreview.net/pdf?id=kOu3-S3wJ7       |
| NET^3^   | WWW        | 2021 | https://dl.acm.org/doi/10.1145/3442381.3449969 |
| PoGeVon  | KDD        | 2023 | https://dl.acm.org/doi/10.1145/3580305.3599444            |

<h2> Run Experiments </h2>
To run experiments and compute metrics, use the `run_imputation.py` script. Here's an example command:
```
 python run_imputation.py --config config/rectsi/air36.yaml
```
When running experiments for PEMS-BA, PEMS-LA and PEMS-SD datasets, one needs to change the subdataset_name value in config file pems.ymal to `PEMS-04`, `PEMS-07` and `PEMS-11` respectively.

## Acknowledgement
This repo is based on the implementations of [GRIN](https://github.com/Graph-Machine-Learning-Group/grin) and [PoGeVon](https://github.com/Derek-Wds/PoGeVon/), thanks for their contribution.
