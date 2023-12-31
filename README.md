# ChromaCert – Certificates for Chromatic Equivalence

(c) 2023 by Harald Bögeholz

This software provided under an MIT license. See the
file [LICENSE](LICENSE) for details.

## Prerequisites

* Download and install the latest Python 3 from https://python.org
  (ChromaCert was developed using Python 3.11 under macOS)
* `pip3 install matplotlib numpy sympy scipy networkx pyqt5`
* Under Windows, it may be required to install the [Microsoft C/C++ 
  runtime libraries](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170).
  (Do that if you get the error `ImportError: DLL load failed while 
  importing _cext` upon program start.)

## Running the program

`python3 chromacert.py` or `python chromacert.py`, depending on your 
system. 

## Documentation

This project is in a very early stage of development. Most of the
functionality is in context menus. Try right-clicking on graphs, row
labels, operands, parentheses ... it is hopefully somewhat self-explanatory
(for someone who knows about chromatic polynomials of graphs).

The [User Guide](docs/userguide.md) is work in progress, as is the 
software.

Click on things labelled "DEBUG" or "Test" at your own risk. Who knows what
they might do today, and they might do something else a few hours later ;-)
. Also, they might crash the program.

Here's a screenshot:

![Screenshot](screenshot-w5.png)

## References

1. Kerri Morgan, Graham Farr: Certificates of Factorisation for Chromatic
   Polynomials. The Electronic Journal of Combinatorics, Volume 16, Issue
   1 (2009). DOI: [10.37236/163](https://doi.org/10.37236/163)
2. Zoe Bukovac, Graham Farr, Kerri Morgan: Short certificates for chromatic
   equivalence, Journal of Graph Algorithms and Applications 2019,
   DOI: [10.7155/jgaa.00490](https://dx.doi.org/10.7155/jgaa.00490)
