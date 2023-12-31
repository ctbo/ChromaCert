# ChromaCert – Certificates for Chromatic Equivalence

(c) 2023 by Harald Bögeholz

This software provided under an MIT license. See the
file [LICENSE](../LICENSE) for details.

**Disclaimer:** This software is a specialised tool for mathematicians
working on chromatic polynomials of graphs. As such, it is not aiming for
optimal usability for a general audience. For the mathematical background,
see the [References](#references) at the end of this document.

**Disclaimer 2:** This software is under development. Beware of clicking on
anything named DEBUG. And use at your own risk, anyway.

## Introduction

This program allows the manipulation of algebraic expressions of graphs,
with the understanding that each graph stands for its chromatic polynomial.
The purpose is the creation of a *certificate for chromatic equivalence*, a
series of transformations that preserve the chromatic polynomial.

The user interface is organised in numbered rows, each containing an
expression. Any transformation or algebraic manipulation will create a new
row with a row label indicating the type of operation. Once a row has been
used to create a new row, it is no longer editable (except for graph
layouts) in order to enforce the correctness of the certificate. This is
indicated by a dark grey background. If you want to try a different
operation on a dark grey row, right-click on the row label on the left and
choose **Append as new Row**. This action creates an editable row,
indicated by a lighter grey background, where you can perform additional
operations.

The software currently does not support deleting rows. If you encounter an
error or wish to alter your approach, simply use **Append as new Row** to
initiate a new sequence of operations. When exporting a certificate to
LaTeX ([see below](#exporting-to-latex)), only the relevant rows forming
part of the certificate will be included.

## Loading and Saving

You can save your work using the **File/Save as ...** menu. Note that the
software currently doesn't remember the current document's name and also
**doesn't give any warning about losing your work**. If you use
**File/New** or **File/Open ...** the new content will immediately replace
the current content.

The file extension used by this software is ".chroma".

## Graph Editing

Upon creating a new graph via the **New Graph** menu, basic graph editing
tools are available. You can select or deselect vertices by clicking on
them. The context menu of the graph offers the options **Create Vertex**
, **Toggle Edge**, and **Delete Vertices**.

**Note:** Editing is permitted only for newly created graphs that are in a
row by themselves. Editing is disabled for rows derived from previous rows,
as indicated by a darker grey background.

## Graph Layout

You can modify the layout of a graph at any time, regardless of the editing
or selection status. Vertices can be moved manually, or you can select from
various layout options available in the context menu of the graph.

The option **Broadcast Layout to Isomorphic** will find all graphs within
the same row that are isomorphic to the graph you clicked on and change
their layout to match the current one.

## Transformations

The following transformations express the chromatic polynomial of one or
more graphs in terms of the chromatic polynomial(s) of other graph(s). This
is where the actual math comes in; see the [References](#references) at the
end of this document

### Addition-Identification

Select exactly two non-adjacent vertices and choose
**Addition-Identification**
from the context menu of the graph. Will create a sum of two graphs where
one has an extra edge and one has the two vertices identified.

### Deletion-Contraction

Select exactly two adjacent vertices and choose **Deletion-Contraction**
from the context menu of the graph. Will create the difference of two
graphs where one has the edge deleted and one has the edge contracted.

### Separate at Clique

For a connected graph, select a clique and choose **Separate at Clique**
from the context menu of the graph. The clique has to be separating, i.e.
after removing the clique the graph must have more than one connected
component. Will create one graph per connected component of the separated
graph, divided by the clique. **Edge case: The zero-clique.** If the graph
is not connected and no vertices are selected, the graph is split into its
connected components.

### Glue at Clique

Select a clique of the same size in two different graphs within the same
product. Then choose **Glue at Clique** from the graph's context menu. This
function merges the two graphs at the selected cliques, *matching the
vertices based on their vertical position*.

**Note:** This function is available only if exactly two graphs in the
entire row have selected vertices.

### Whitney Flip

In a connected graph, select two vertices whose removal results in the
graph splitting into exactly two connected components. Choose **Whitney
Flip** from the graph's context menu to execute the operation, which
involves cutting the graph at the selected vertices and reassembling it
with one segment flipped.

### Disjoint Union

Right-click on the × operator between two graphs and choose **Disjoint
Union** from the context menu. Will replace the two graphs by their
disjoint union as a single graph.

## Algebraic manipulations

This software is designed for manually manipulating expressions and
performs minimal automatic simplifications.

Each expression is either a sum or a product of sub-expressions or graphs
where sums are nested inside products inside sums and so forth. By default,
only necessary parentheses are shown. Select
**View/Full Expression Structure** from the main menu to display the full
structure. Round parentheses are used around sums and square brackets are
used around products. This feature is particularly useful for accessing the
context menu of an otherwise invisible open bracket.

### Collecting and splitting terms

To consolidate multiple isomorphic graphs within the same sum or product
into a single term, choose **Collect Isomorphic** from the context menu.
The opposite operation is called **Split Term**. It will split off one copy
of the graph as a separate summand or factor, depending on context.

### Changing the order of terms

Each binary operator (+, –, ×) has the option **Flip** in its context menu.
It will swap the summands/factors immediately surrounding the operator.

### Inserting neutral expressions

It can sometimes be useful to introduce a helper graph G into an
expression. This can be done as a zero sum (+ G – G) or as a neutral
product (× G / G). To prepare the graph G, first place it on a separate
row. You can either

* select a graph from the **New Graph** main menu

or

* right-click any graph and select **Copy as new Row** from the context
  menu.

In both cases, the new graph will be editable to suit your needs. Let's
assume, for example, that you have created a new graph in row (7). In the
expression you want to manipulate, right-click on any graph or on any open
parenthesis "(" or bracket "[" and select **Insert Neutral**
from the context menu. Depending on whether the expression immediately
surrounding the object you clicked on is a sum or a product, you will be
offered "+ (7) – (7)" or "× (7) / (7)" in a sub-menu, along with similar
options for any other isolated graphs in the document. The neutral
expression will be inserted *before* the item you clicked on.

### Adding and removing brackets

To insert neutral terms accurately or to control the application of the
distributive law, you may need to add extra brackets to your expressions.

* Right-click on a graph and choose **Add Brackets** to add an extra pair
  of brackets around a single graph.

* If you want to add brackets around a sub-expression within a sum or
  product, right-click on the *leftmost* binary operator (+, –, ×)
  you want to include in the brackets and choose **Add Brackets**. A
  sub-menu will allow you to choose how many summands/factors to include in
  the brackets.

ChromaCert will automatically turn on the **View/Full Expression
Structure** option to display newly inserted brackets that would otherwise
be invisible.

Once you have completed your algebraic manipulations, you can simplify the
expression by removing any unnecessary brackets. To do this, right-click on
the row label and select **Simplify Brackets**. It may also be helpful to
deselect the **View/Full Expression Structure** option in the main menu to
return to the default view.

### Applying the distributive law

If you right-click a graph that is a factor in a product within a sum, the
context menu item **Factor Out** will become available. The clicked graph
will be factored out of the sum, adjusting the other summands as necessary.

If you right-click a graph that is a factor in a product and that product
also has a sum to the right of the selected factor, the context menu item
**Distribute Right** will become available. The clicked graph will be
distributed over the next sum encountered to the right.

## Exporting to LaTeX

To prepare a certificate for publication, right-click on the row label of
the *last row* and choose one of the two **Copy as LaTeX** options.
Option **(L)** places the explanation in the left margin whereas option
**(R)** places it to the right of the equations. The certificate is copied
to the clipboard, ready for inserting it into your favourite LaTeX editor.

Note: All the rows in this software form a directed acyclic graph with each
row having at most one parent. The LaTeX export follows the chain of
parents to the beginning and then generates the derivation of the current
row. That's why the ability to delete rows has not been a priority in
developing this software — the LaTeX export just ignores superfluous rows.

Acknowledging the use of this software in any publications would be
appreciated.

## References

1. Kerri Morgan, Graham Farr: Certificates of Factorisation for Chromatic
   Polynomials. The Electronic Journal of Combinatorics, Volume 16, Issue
   1 (2009). DOI: [10.37236/163](https://doi.org/10.37236/163)
2. Zoe Bukovac, Graham Farr, Kerri Morgan: Short certificates for chromatic
   equivalence, Journal of Graph Algorithms and Applications 2019,
   DOI: [10.7155/jgaa.00490](https://dx.doi.org/10.7155/jgaa.00490)
