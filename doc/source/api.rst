.. _api:

API Reference
-------------


Nexus Backend
^^^^^^^^^^^^^

This backend submits Qibo circuits through Quantinuum Nexus. Configure the
desired target using the ``platform`` argument, for example
``"hseries:H2-1LE"``, ``"helios:Helios-1E"``, or ``"aer:aer_simulator"``.

.. note::
   Importing the package does not trigger Nexus authentication. Authentication
   and project resolution happen lazily on the first execution or estimation
   call.

.. autoclass:: nexus.backend.NexusClientBackend
    :members:
    :member-order: bysource


MetaBackend
^^^^^^^^^^^

.. autoclass:: nexus.MetaBackend
    :members:
    :member-order: bysource
