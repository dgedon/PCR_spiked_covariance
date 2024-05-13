# No Double Descent in Principal Component Regression

This is the official repository for the paper "No Double Descent in Principal Component Regression: A High-Dimensional Analysis"
by Daniel Gedon, Antônio H. Ribeiro and Thomas B. Schön. which is presented at ICML 2024.

## Paper summary

In this paper, we study the double descent phenomenon in the context of Principal Component Regression (PCR).
We utilise the spiked covariance model to generate the data and derive the asymptotic risk of PCR using random matrix theory.
The analysis is also extended to distribution shift scenarios.

## Requirements

The code has been tested for Python 3.7.4. The notebook requires
- numpy
- matplotlib
- scipy
- tqdm

## Demo

We provide a simple demo in a Jupyter notebook. 
This notebook contains:
- spiked covariance data generation
- PCR model in simulation
- PCR model with analytical results
- Baseline: Full regression simulation
- Baseline: Full regression with analytical results

## Citation

If you use this code in your research or find it helpful, please consider citing the paper:
```bibtex
@inproceedings{gedon2024pcr,
  title={No Double Descent in Principal Component Regression: A High-Dimensional Analysis},
  author={Gedon, Daniel and Ribeiro, Ant{\^o}nio H and Sch{\"o}n, Thomas B},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```