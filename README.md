# GraphSAGE
This is a PyTorch implementation of GraphSAGE from the paper [Inductive Representation Learning on Large Graphs](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs).

## Usage

In the `src` directory, edit the `config.json` file to specify arguments and
flags. Then run `python main.py`.

## Limitations
* Currently, only supports the Cora dataset. However, for a new dataset it should be fairly straightforward to write a Dataset class similar to `datasets.Cora`.

## References
* [Inductive Representation Learning on Large Graphs](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs), Hamilton et al., NeurIPS 2017.
* [Collective Classification in Network Data](https://www.aaai.org/ojs/index.php/aimagazine/article/view/2157), Sen et al., AI Magazine 2008.