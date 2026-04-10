.. currentmodule:: pytket.extensions.aqt

Changelog
~~~~~~~~~

unreleased
----------

* Add depth block analysis
* Add hypergraph partitioning based gate selection

0.40.0 (January 2026)
---------------------

* Updated pytket version requirement to 2.11.

0.38.0 (November 2025)
----------------------

* New greedy gate selection algorithm
* Updated pytket version requirement to 2.5.

0.37.0 (June 2025)
------------------

* fix validation error (barrier treated as gate)
* drop Python 3.10
* Update pytket minimium version requirement to 2.5.0.
* Respect memory and gate zones during routing
* Update to mt-kahypar 1.5
* Update poetry minimum version requirement to 2.0.0

0.36.0 (November 2024)
----------------------

* Updated pytket version requirement to 1.35.

0.35.0 (October 2024)
---------------------

* Updated pytket version requirement to 1.33.
* Add graph partition based routing for AQTMultiZoneBackend

0.34.0 (August 2024)
--------------------

* Implement new aqt api
* Fix handling of ``simplify_initial`` kwarg in ``process_circuits()``.
* Updated Qiskit AQT Provider to v1.8.
* Updated pytket version requirement to 1.31.

0.33.0 (March 2024)
-------------------

* Updated pytket version requirement to 1.26.

0.32.0 (January 2024)
---------------------

* Updated pytket version requirement to 1.24.
* Python 3.12 support added

0.31.0 (January 2024)
---------------------

* Updated pytket version requirement to 1.23.

0.30.0 (November 2023)
----------------------

* Updated pytket version requirement to 1.22.

0.29.0 (October 2023)
---------------------

* Add simple automatic placement and routing for multi-zone architectures
* Don't include ``SimplifyInitial`` in default passes; instead make it an option
  to ``process_circuits()``.
* Updated pytket version requirement to 1.21.

0.28.0 (June 2023)
------------------

* Updated pytket version requirement to 1.16.

0.27.0 (May 2023)
-----------------

* Updated pytket version requirement to 1.15.

0.26.0 (April 2023)
-------------------

* Change default optimisation level to 2
* Add Rz gate to aqt backend
* Add experimental support for multi zone ion trap architectures
* Dropped support for python 3.8 and 3.9, added support for 3.11
* Updated pytket version requirement to 1.14.

0.25.0 (November 2022)
----------------------

* Updated pytket version requirement to 1.9.

0.24.0 (October 2022)
---------------------

* Updated pytket version requirement to 1.7.

0.23.0 (August 2022)
--------------------

* Updated pytket version requirement to 1.5.

0.22.0 (July 2022)
------------------

* Updated pytket version requirement to 1.4.

0.21.0 (June 2022)
------------------

* Updated pytket version requirement to 1.3.

0.20.0 (May 2022)
-----------------

* Updated pytket version requirement to 1.2.

0.19.0 (April 2022)
-------------------

* Updated pytket version requirement to 1.1.

0.18.0 (March 2022)
-------------------

* Updated pytket version requirement to 1.0.

0.17.0 (February 2022)
----------------------

* Updated pytket version requirement to 0.19.
* Drop support for Python 3.7; add support for 3.10.

0.16.0 (January 2022)
---------------------

* Updated pytket version requirement to 0.18.

0.15.0 (November 2021)
----------------------

* Updated pytket version requirement to 0.17.

0.14.0 (October 2021)
---------------------

* Updated pytket version requirement to 0.16.

0.13.0 (September 2021)
-----------------------

* Updated pytket version requirement to 0.15.

0.12.0 (September 2021)
-----------------------

* Updated pytket version requirement to 0.14.

0.11.0 (July 2021)
------------------

* Updated pytket version requirement to 0.13.

0.10.0 (June 2021)
------------------

* Updated pytket version requirement to 0.12.


0.9.0 (May 2021)
----------------

* Updated pytket version requirement to 0.11.

0.8.0 (May 2021)
----------------

* Contextual optimisation added to default compilation passes (except at optimisation level 0).

0.7.0 (April 2021)
------------------

* Fix to rebase pass.
