Getting Started
===============

Installation
------------

.. code-block:: bash

   pip install -e .

Building a Classifier
---------------------

.. code-block:: python

   pipeline = ZeroShotTopicPipeline("deepset/sentence_bert")
   pipeline.add_labels(labels)

.. code-block:: python

   pipeline.add_projection_matrix(projection_matrix)

