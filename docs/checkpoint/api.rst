API reference
=============

Store
-----

.. autoclass:: graphed_checkpoint.Store
   :members:

.. autoclass:: graphed_checkpoint.JournalEntry
   :members:

Resumable runner
----------------

.. autofunction:: graphed_checkpoint.run_resumable

.. autoclass:: graphed_checkpoint.ResumeResult
   :members:

.. autoclass:: graphed_checkpoint.ResumeReport
   :members:

Retry policies
--------------

.. autoclass:: graphed_checkpoint.RetryN
   :members:

.. autoclass:: graphed_checkpoint.RetrySmallerChunk
   :members:

.. autoclass:: graphed_checkpoint.RetryElsewhere
   :members:

.. autoclass:: graphed_checkpoint.Quarantine
   :members:

Error harvesting + codecs
-------------------------

.. autofunction:: graphed_checkpoint.dead_letter_descriptor

.. autoclass:: graphed_checkpoint.PickleCodec
   :members:

.. autoclass:: graphed_checkpoint.NumpyCodec
   :members:
