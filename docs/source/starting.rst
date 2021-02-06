Getting Started
===============

Installation
------------

.. code-block:: bash

   git clone https://github.com/pbmstrk/zeroshot.git
   cd zeroshot
   pip install .

Building a Pipeline
---------------------

.. code-block:: python

   from transfomers import AutoModel, AutoTokenizer
   from zeroshot import ZeroShotPipeline

   tokenizer = AutoTokenizer.from_pretrained("deepset/sentence_bert") 
   model = AutoModel.from_pretrained("deepset/sentence_bert")

   pipeline = ZeroShotPipeline(tokenizer, model)
   pipeline.add_labels(labels)

To improve performance, it is possible to include a projection matrix. For more info, see :ref:`here<projection:get_projection_matrix>`.

.. code-block:: python

   pipeline.add_projection_matrix(projection_matrix)

Example
-------

.. code-block:: python

   import torch
   from transformers import AutoTokenizer, AutoModel
   from zeroshot import ZeroShotPipeline

   tokenizer = AutoTokenizer.from_pretrained("deepset/sentence_bert")
   model = AutoModel.from_pretrained("deepset/sentence_bert")

   phrase = "Who are you voting for in 2020?"
   labels = ['Politics', "Sports", "Fashion"]

   pipeline = ZeroShotPipeline(tokenizer, model)
   pipeline.add_labels(labels)

   predictions = pipeline(phrase)
   print(f"The phrase is about: {labels[torch.argmax(predictions)]}")
   # This phrase is about: Politics