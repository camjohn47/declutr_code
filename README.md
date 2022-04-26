# declutr_code
Developing tools for self-supervised code embeddings using variants of the DeClutr approach: https://arxiv.org/abs/2006.03659. Software is written in Python and neural networks are implemented with Tensorflow. Currently running experiments and building results from CodeSearchNet data (https://github.com/github/CodeSearchNet) across different programming languages. Project's main goal is quick summarization of code semantics with vectors. This is a code embedding framework. 

The motivation is providing pre-processing, code parsing, and Tensorflow models for NLP and sequence modeling, with an emphasis on methods and scripts from programming languages. A new approach for faster masked language modeling is in development. This is called masked method modeling (MMM): a programming version of masked language modeling (MLM) with a significantly reduced vocabulary. Specifically, it restricts the MLM learning objective used in BERT, GPT-2 and other embedding models to method definitions and method calls. The intent is to focus on learning more important tokens (methods) that represent entire segments of code, while reducing runtime and memory usage. 

This project is recent and **still in development**. For more info on how to get started, please check out the notebooks! 

