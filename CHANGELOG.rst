Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.


Version 1.1.0 (2022/12/02)
--------------------------

* Added: ``audmath.rms()``
  to calculate root mean square of signal
* Added: ``audmath.db()``
  to convert value to decibel
* Added: ``audmath.invert_db()``
  to convert value from decibel
* Added: ``audmath.window()``
  to provide different kind
  of (half-)windows 
* Added: support for Python 3.10


Version 1.0.0 (2022/01/03)
--------------------------

* Added: Python 3.9 support
* Removed: Python 3.6 support


Version 0.9.4 (2021/10/25)
--------------------------

* Fixed: bottom margin in API table


Version 0.9.3 (2021/10/25)
--------------------------

* Changed: use new ``sphinx-audeering-theme``


Version 0.9.2 (2021/07/30)
--------------------------

* Fixed: package name in installation docs


Version 0.9.1 (2021/07/29)
--------------------------

* Added: benchmarks for ``audmath.inverse_normal_distribution()``
  against ``scipy``
* Changed: implement ``audmath.inverse_normal_distribution()``
  in a native vectorized way
* Fixed: missing links in changelog


Version 0.9.0 (2021/07/28)
--------------------------

* Added: Initial release
* Added: ``audmath.inverse_normal_distribution()``


.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
