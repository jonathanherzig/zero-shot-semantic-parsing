
## Decoupling Structure and Lexicon for Zero-Shot Semantic Parsing

### Jonathan Herzig and Jonathan Berant
Code for the zero-shot semantic parser described in our EMNLP 2018 [paper](https://arxiv.org/pdf/1804.07918.pdf).

The structure mapper implementation is an extension of [this code](https://worksheets.codalab.org/worksheets/0x50757a37779b485f89012e4ba03b6f4f/). 

#### Setup
1. Install [Miniconda2](https://conda.io/miniconda.html)
1. Install Stanford CoreNLP:
```bash
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
$ unzip stanford-corenlp-full-2016-10-31.zip 
```
3. Install python dependencies:
```bash
$ conda install --file reqs_conda.txt
$ pip install -r reqs_pip.txt
```

#### Preprocess
To delexicalize data for all domains run and prepare cross domain splits use:
```bash
$ python src/py/zero_shot/preprocess.py 
```

#### Train and evaluate
To run one of the models implemented in the paper use:
```bash
$ sh scripts/MODEL.sh SPLIT
```
Where:
`MODEL` is one of the following: `zero_shot, cross_lex, cross_lex_rep, in_abstract, in_lex`.
`SPLIT` is either `test` (the original train/test split of the OVERNIGHT dataset), or `dev` (in this case the original train set is split to 80%/20% train/test sets).  

To run all models use:
```bash
$ sh scripts/run_all.sh
```

Results are saved to `/res` folder. To print all results use:
```bash
$ python src/py/zero_shot/print_res.py
```
