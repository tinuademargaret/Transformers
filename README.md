# Transformer

The transformer architecture introduced in this [paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) implemented from scratch using pytorch out of pure curiosity for how it works.
## Installation and use

If you'd like to try it out, clone this repository

Install [poetry](https://python-poetry.org/docs/#installation) for dependency management

Change into the root directory of this project and then run

```
poetry shell 
```

Then, from the same directory, run:

```
python main.py
```

This would use some predefined tokens in the `main.py` file as input to the transformer network and print out the output from the network.
I plan to implement a full generative model. I'll update this Readme once I do so.

## Reference
- [Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Transformers from scratch](https://peterbloem.nl/blog/transformers) by Peter Bloem which I took so much delight in reading
- [Transformers from scratch](https://e2eml.school/transformers.html) by Brandon Roher, another lovely article.