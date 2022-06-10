# declutr_code
Developing tools for self-supervised code embeddings using variants of the [DeClutr approach](https://arxiv.org/abs/2006.03659). Software is written in Python and neural networks are implemented with Tensorflow. Currently running experiments and building results from the wonderful [CodeSearchNet](https://github.com/github/CodeSearchNet) project across different programming languages. Project's main goal is quick, effective representation and summarization of what code segments do. This is a code embedding framework with applications ranging from small method representation to script summarization. 

The motivation is providing pre-processing, code parsing, and Tensorflow models for NLP and sequence modeling, with an emphasis on methods and scripts from programming languages. Experimenting with a new approach for faster masked language modeling called masked method modeling (MMM). MMM is a programming version of masked language modeling (MLM) with a significantly reduced vocabulary. It restricts the MLM learning objective used in BERT, GPT-2 and other embedding models to method definitions and method calls. The intent is to focus on learning more important tokens (methods) that represent entire segments of code, while reducing runtime and memory usage. 

# Installation 
In addition to the standard git clone command, you need to add the recursive flag so that CodeSearchNet module is added as well. You can copy and paste the command below.

```bash
git clone https://github.com/camjohn47/declutr_code --recursive
```
Then, please run the "setup.py" script in the main directory to set up path dependencies. After doing this, you should be
able to run any script in the "scripts" directory. 

**This project is new and still in active development**. Note that you don't have to use CodeSearchNet data. The CodeParser can be used to parse through scripts in any accessible directory and organize them for training data. For more info on how to get started, please check out the notebooks! 

