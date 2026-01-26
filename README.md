# Self-Aware Knowledge Probing
<hr>
This repository contains the code, data and results of the paper:

***Self-Aware Knowledge Probing: Evaluating Language Models’ Relational Knowledge through Confidence Calibration***  

by Christopher Kissling, Elena Merdjanovska and Alan Akbik

## Overview
<hr>

*Knowledge probing* quantifies how much relational knowledge a language model has acquired
during pre-training. Existing knowledge probes evaluate model capabilities through metrics
like prediction accuracy and precision. Such evaluations fail to account for the model’s
reliability, reflected in the calibration of its *confidence scores*.

We propose a *knowledge calibration probing* framework that decomposes model confidence
into three distinct modalities:

1. ***Intrinsic confidence***: derived from raw log-probabilities.  
2. ***Structural consistency***: measured through agreement across semantically equivalent rephrasings.  
3. ***Semantic grounding***: evaluated through the model’s response to explicit linguistic markers of uncertainty. 

Experiments are conducted in a closed-set probing setup based on the BEAR framework
[(Wiland et al., 2024)](https://arxiv.org/abs/2104.07554) and cover both causal and masked language models.

This repository serves as a **research artifact** supporting the experiments reported in the paper.

## Repository Structure
<hr>

```
├── data/            # datasets and prompt templates
├── src/             # probing and calibration code
│   ├── estimates.py
│   ├── main.py
│   ├── metrics.py
│   └── utils.py
├── scores/          # model scores
├── results/
│   └── notebooks/   # notebooks with our results
├── requirements.txt
└── README.md
```

## Usage
<hr>

A typical workflow of self-aware knowledge probing of a language model consists of:

1. Run `src/main.py` for an LM available in the *Hugging Face Hub*. Specify the template
indices (see below for an overview) and model type. The raw (instance-level)
scores will be saved to `scores/BEAR/model/scores.json`.
2. Call the function `get_confidences_estimates()` from `src/estimates.py`. It returns a
pandas dataframe with instance-level correctness and confidences for the available estimates.
3. Choose a metric from `src/metrics.py` to evaluate calibration. You
need to pass the correctness labels and confidences.

### Template indices and confidence modalities
1. ***Intrinsic Confidence***: derived from single template predictions,
i.e. any of the indices `[0, 1, 2, 3, 4]`.
2. ***Structural Consistency***: obtained from all semantically equivalent templates, i.e.
the predictions from all templates `[0, 1, 2, 3, 4]`.
3. ***Semantic Grounding***:
   1. **Verbalized**: template index `5` for *possibly*, `6` for *certainly*.
   2. **Numerical**: template indices `[7, 8, 9, 10, 11]`.

See section three of the paper for a more detailed explanation. The templates for each relation
can be found in `data/BEAR/metadata_relations.json`.

## Citation
<hr>

If you use this code or the proposed framework in your work, please cite:
```
@article{kissling2025selfawareknowledgeprobing,
  title={Self-Aware Knowledge Probing: Evaluating Language Models’ Relational Knowledge through Confidence Calibration},
  author={Kissling, Christopher and Merdjanovska, Elena and Akbik, Alan},
  year={2025}
}
```