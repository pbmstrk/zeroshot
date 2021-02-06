Zero-Shot Classification
========================

Zero-shot text classification by measuring similarities in an embedding space. The method used is described in this  `blogpost <https://joeddav.github.io/blog/2020/05/29/ZSL.html>`_ (along with other approaches to zero-shot learning). Experiments are run on the dataset presented in this `paper <https://www.aclweb.org/anthology/D19-1404/>`__ and models used are presented in  this `paper <https://www.aclweb.org/anthology/D19-1410/>`__.

In short, BERT is used to encode sentence and this representation is compared to the encoding og the labels, see *Fig. 1*. The label with the highest similarity to the sentence encoding is selected.

.. figure:: _images/sim-1.png
   :height: 400
   :align: center

   *Fig. 1*: Overview of architecture

Results
-------

The results obtained on the dataset presented `here <https://www.aclweb.org/anthology/D19-1404/>`__ are given in the table below.

.. csv-table:: Results


   **model**, **projection matrix**, **k**, **lambda**, **score**
   deepset/sentence_bert, \-, \-, \-, 37.743
   deepset/sentence_bert, Word2Vec, 5000, 5, 43.398
   deepset/sentence_bert, GloVe, 20000, 10, **47.740**


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
   projection


