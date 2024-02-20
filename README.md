# Citation Amnesia: NLP and Other Academic Fields Are in a Citation Age Recession

[![arXiv](https://img.shields.io/badge/arXiv-2402.12046-b31b1b.svg)](https://arxiv.org/abs/2402.12046)

## The Repository

This repository contains the code and datasets used in the paper "Citation Amnesia: NLP and Other Academic Fields Are in a Citation Age Recession". Our work analyzes citation patterns across various academic fields to investigate trends in citation ages, highlighting the growing focus on recent literature at the expense of older, foundational works.


## Repository Structure

- `data/`: Contains datasets by field, including citation relationships, grant information, institutional affiliations, and more.
- `figures/`: Directory containing generated figures and plots.
- `output/`: Contains generated tables and other outputs from the analysis.
- `analysis.ipynb`: Jupyter notebook with main plots and analysis from the paper.
- `helpers.py`: Python module with functions for calculations and postprocessing.
- `dataset.py`: Python script for downloading and handling the dataset.

## Getting Started

To replicate the analysis or explore the datasets, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed. A list of dependencies can be found in `requirements.txt`.
3. Run `dataset.py` to download and prepare the datasets.
4. Use `helpers.py` for specific calculations and data postprocessing tasks.
5. Open and execute the `analysis.ipynb` notebook for visualizations and further analysis.

## Usage

### Data Download and Preparation

To download and prepare the dataset, run:

```bash
python dataset.py
```

This script will download the necessary data files and prepare them for analysis.

### Calculations and Postprocessing

`helpers.py` contains functions used for various calculations and data processing tasks. These functions are used within the analysis notebook but can also be imported and used in other scripts.

### Analysis and Visualization

Open `analysis.ipynb` in a Jupyter environment with R (for example with GitHub Codespaces) to see the key plots and analysis performed in the study.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use the code or data from this repository in your research, please cite our paper:

```bibtex
@article{wahle2024citationamnesia,
  title={Citation Amnesia: NLP and Other Academic Fields Are in a Citation Age Recession},
  author={Wahle, Jan Philip and Ruas, Terry and Abdalla, Mohamed and Gipp, Bela and Mohammad, Saif M.},
  journal={arXiv preprint arXiv:2402.12046},
  year={2024}
}
```

## Contributing

Contributions to this project are welcome! Please submit issues and pull requests with any suggestions, corrections, or enhancements.