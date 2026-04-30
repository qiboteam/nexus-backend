.. nexus-backend documentation master file

Welcome to nexus-backend
========================

A Qibo backend for compiling and executing circuits through Quantinuum Nexus.

Install with::

   pip install git+https://github.com/qiboteam/nexus-backend

Then in Python::

   import qibo

   qibo.set_backend("nexus", platform="hseries:H2-1LE")
   circuit = qibo.models.QFT(5)
   circuit.add(qibo.gates.M(0, 2, 4))
   result = circuit(nshots=1000)
   print(result.frequencies())

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
