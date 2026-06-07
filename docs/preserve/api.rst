API reference
=============

Bundle
------

.. autofunction:: graphed_preserve.build_bundle

.. autofunction:: graphed_preserve.reproduce

.. autofunction:: graphed_preserve.inspect

.. autoclass:: graphed_preserve.Bundle
   :members:

Manifest + fingerprint
----------------------

.. autofunction:: graphed_preserve.fingerprint

.. autofunction:: graphed_preserve.canonical_bytes

External payload plugins
------------------------

Externals are an extensible plugin system: each plugin gives a ``kind`` a deterministic,
content-based ``content_hash`` (ONNX hashes its weights, correctionlib its contents) and an
``evaluate``. ``register_plugin`` validates the hash for cross-process determinism and non-vacuity.
``onnx`` and ``correctionlib`` ship as plugins and double as templates for users' own Externals.

.. autoclass:: graphed_preserve.ExternalPlugin
   :members:

.. autoclass:: graphed_preserve.ResourceCache
   :members:

.. autofunction:: graphed_preserve.register_plugin

.. autofunction:: graphed_preserve.validate_plugin

.. autofunction:: graphed_preserve.record_external

.. autofunction:: graphed_preserve.get_plugin

.. autofunction:: graphed_preserve.registered_kinds

.. autofunction:: graphed_preserve.evaluate_external

.. autofunction:: graphed_preserve.sha256_bytes

Errors
------

.. autoclass:: graphed_preserve.PreserveError
   :members:

.. autoclass:: graphed_preserve.UnresolvedPayload
   :members:
