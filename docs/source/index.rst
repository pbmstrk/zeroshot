Zero-Shot Classification
========================

Introduction
------------

Inspired by this `blogpost <https://joeddav.github.io/blog/2020/05/29/ZSL.html>`_, this `paper <https://www.aclweb.org/anthology/D19-1404/>`__ on zero-shot text classification and this `paper <https://www.aclweb.org/anthology/D19-1410/>`__ on sentence embeddings.

.. figure:: _images/bert-1.png
   :height: 400
   :align: center

   *Fig. 1*: Overview of BERT


.. figure:: _images/sim-1.png
   :height: 400
   :align: center

   *Fig. 2*: Overview of architecture


Results
-------

.. csv-table:: Results


   **model**, **projection matrix**, **k**, **lambda**, **score**
   deepset/sentence_bert, \-, \-, \-, 37.743
   deepset/sentence_bert, Word2Vec, 5000, 5, 43.398
   deepset/sentence_bert, GloVe, 20000, 10, 47.740


.. toctree::
   :hidden:
   :maxdepth: 2
   
   self

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Package Documentation:

   starting
   data
   vec
   pipeline


