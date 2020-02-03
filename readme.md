## About

This is the repository for the paper Neural Topic Model with Attention for Supervised Learning, to appear at AISTATS 2020.

## Requirements

* Python 3.6.5
* Tensorflow 1.14.0
* NLTK

## Usage

* The 20 newsgroups dataset is obtained from [https://github.com/alexeygrigorev/datasets/tree/master/20-newsgroups/preprocessed](https://github.com/alexeygrigorev/datasets).

* To train a Topic Attention Model (TAM) with 50 topics:
```
python train_topical.py --num_topics 50
```
The default vocabulary size is 2000. Can be changed by selecting your own vocabulary. For other changable hyperparameters, see train_topical.py. train_vanilla.py is for the Attention-RNN baseline. train_topic_model.py is for the GSM baseline. They are of the similar usage.

## Citation

```
@inproceedings{TAM_2020,
	author = {Xinyi Wang and Yi Yang}, 
	title = {Neural Topic Model with Attention for Supervised Learning}, 
	booktitle = {Proceedings of AISTATS},
	year = {2020}
}
```