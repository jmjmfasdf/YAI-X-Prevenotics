# YAI-X-Prevenotics
<!-- HEADER START -->
<!-- src: https://github.com/kyechan99/capsule-render -->
<p align="center"><a href="#">
    <img width="100%" height="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:C0C0C0,50:87CEEB,100:1E90FF&height=220&section=header&fontSize=40&fontColor=ffffff&animation=fadeIn&fontAlignY=40&text=Personalized%20EGD%20Report%20Generation" alt="header" />
</a></p>

<!-- HEADER END -->

# Members 👋
<b> <a href="https://github.com/rubato-yeong">김진영</a></b>&nbsp; :&nbsp; YAI 13th&nbsp; /&nbsp; jinyeong1324@yonsei.ac.kr<br>
<b>  <a href="https://github.com/jmjmfasdf">서정민</a></b>&nbsp; :&nbsp; YAI 12th&nbsp; /&nbsp; jmme425@yonsei.ac.kr  <br>
<b> <a href="https://github.com/0914eagle">전희재</a></b>&nbsp; :&nbsp; YAI 12th&nbsp; /&nbsp; 0914eagle@yonsei.ac.kr <br>
<b> <a href="#">고현아</a></b>&nbsp; :&nbsp; YAI 13th&nbsp; /&nbsp; kha9867@yonsei.ac.kr <br>
<b> <a href="https://github.com/1n1ng">조인희</a></b>&nbsp; :&nbsp; YAI 13th&nbsp; /&nbsp; choinh@yonsei.ac.kr <br>

<br>

# Getting Started 🔥
This project is based on model from [Med42-Llama](https://huggingface.co/m42-health/Llama3-Med42-70B). We would like to acknowledge the contributors and maintainers of that repository for their valuable work.

## Setup
### GPU Installation (Optional)

To enable GPU support, install TensorFlow with CUDA by running:

```bash
pip install tensorflow[and-cuda]
```

### Installation Steps 
To set up desired environment, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/jmjmfasdf/YAI-X-Prevenotics.git
```

2. Install conda environment:
```bash
conda create -n prevenotics python=3.11
conda activate prevenotics
```

3. Install python requriements:
```bash
pip install -r requirements.txt
```

4. Install pretrained model:
```bash
python download_model.py
```

## Usage 

### Report generation

This script calculates the average similarity between multiple people. If a save path is not specified, it plots the similarity matrix; otherwise, it saves the similarity matrix to the specified path.

You can manually run python code individually by running:
```bash
python run_qa.py
```

<br>

# Skills
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> 

<br>

# Citations

### Models
```bibtex
@misc{2408.06142,
Author = {Clément Christophe and Praveen K Kanithi and Tathagata Raha and Shadab Khan and Marco AF Pimentel},
Title = {Med42-v2: A Suite of Clinical LLMs},
Year = {2024},
Eprint = {arXiv:2408.06142},
}
```

### Datasets
```bibtex
@inproceedings{hartvigsen2022toxigen,
  title={ToxiGen: A Large-Scale Machine-Generated Dataset for Implicit and Adversarial Hate Speech Detection},
  author={Hartvigsen, Thomas and Gabriel, Saadia and Palangi, Hamid and Sap, Maarten and Ray, Dipankar and Kamar, Ece},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}

@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}

```



