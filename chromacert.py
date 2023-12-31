# ChromaCert (c) 2023 by Harald Bögeholz

# This software provided under an MIT license. See the file LICENSE for details.

from __future__ import annotations
from typing import Tuple

import sys
from copy import deepcopy
import math
import itertools
import json
import traceback

import networkx as nx
from networkx import NetworkXException
import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QLabel, QLayout, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QHBoxLayout, QSizePolicy
from PyQt5.QtWidgets import QMenu, QActionGroup
from PyQt5.QtGui import QPalette, QPixmap
from PyQt5.QtCore import Qt, QSize

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)

import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 0

# configure multiplication symbol according to taste
# TIMES_SYMBOL = "·"
TIMES_SYMBOL_LATEX = r"\cdot"
TIMES_SYMBOL = "×"
# TIMES_SYMBOL_LATEX = r"\times"
ROW_BACKGROUND_LIGHT = "#F0F0F0"
ROW_BACKGROUND_MEDIUM = "#E0E0E0"
ROW_BACKGROUND_DARK = "#A0A0A0"


def clear_layout(layout):
    if isinstance(layout, QLayout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                clear_layout(item.layout())


def pretty_minus(s):
    return s.replace("-", "–")


def layout_with_scaffold(graph, layout_func=nx.spring_layout):
    """
    Layout a graph using a given layout function, but first make sure the graph is connected by picking
    a minimum degree node from each component and connecting those with a clique.
    When using spring_layout, this should prevent the components of a graph from drifting apart
    :param graph: a graph
    :param layout_func: an optional layout function
    :return: a dictionary with nodes as keys and positions as values
    """
    components = list(nx.connected_components(graph))
    if len(components) <= 1:
        return layout_func(graph)

    scaffold_nodes = [min(comp, key=lambda node: graph.degree(node)) for comp in components]
    scaffolded_graph = graph.copy()
    # connect components by forming a clique between scaffold_nodes
    for i in range(len(scaffold_nodes)):
        for j in range(i+1, len(scaffold_nodes)):
            scaffolded_graph.add_edge(scaffold_nodes[i], scaffold_nodes[j])

    return layout_func(scaffolded_graph)


class GraphHash:
    """
    A Weisfeiler–Lehman hash of a graph. This class keeps track of all hashes ever created.
    The __repr__ method returns a shortened hash with the minimum number of digits that still make it unique
    among all hashes created so far.
    """
    all_hashes = set()

    def __init__(self, graph):
        self.hash = nx.weisfeiler_lehman_graph_hash(graph)
        self.all_hashes.add(self.hash)

    def __repr__(self):
        for unique_digits in range(1, 33):
            if len(self.all_hashes) == len({h[-unique_digits:] for h in self.all_hashes}):
                return self.hash[-unique_digits:]
        assert False, "This can't happen."

    def __eq__(self, other):
        if isinstance(other, GraphHash):
            return self.hash == other.hash
        return False


class RowLabel(QLabel):
    def __init__(self, row=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row = row
        font = self.font()
        font.setBold(True)
        self.setFont(font)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        context_menu.setStyleSheet(f"background-color: {default_menu_bg_color_name};")

        simplify_action = context_menu.addAction("Simplify Brackets")
        simplify_action.triggered.connect(self.on_simplify)

        copy_action = context_menu.addAction("Append as new Row")
        copy_action.triggered.connect(self.on_copy)

        latex_action = context_menu.addAction("Copy as LaTeX (L)")
        latex_action.triggered.connect(self.on_latex)

        latex_action = context_menu.addAction("Copy as LaTeX (R)")
        latex_action.triggered.connect(self.on_latex_right)

        debug_action = context_menu.addAction("DEBUG Row")
        debug_action.triggered.connect(self.on_debug)

        debug2_action = context_menu.addAction("DEBUG background")
        debug2_action.triggered.connect(self.on_debug2)

        debug3_action = context_menu.addAction("DEBUG layout")
        debug3_action.triggered.connect(self.on_debug3)

        # Display the context menu
        context_menu.exec(event.globalPos())

    def on_simplify(self):
        self.row.simplify_nesting()

    def on_copy(self):
        self.row.copy_as_new_row()

    def on_latex(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.row.derivation_to_latex_raw())

    def on_latex_right(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.row.derivation_to_latex_raw(True))

    def on_debug(self):
        print(f"({self.row.row_index+1}): {self.row.graph_expr}")

    def on_debug2(self):
        self.row.set_background_color()

    def on_debug3(self):
        for i in range(self.row.main_window.main_layout.count()):
            item = self.row.main_window.main_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                print(f"Widget: {type(widget)}")
            else:
                child_layout = item.layout()
                if child_layout is not None:
                    print(f"Child Layout: {type(child_layout)}")
                else:
                    print("SpacerItem or other non-widget, non-layout item")


class OpLabel(QLabel):
    def __init__(self, op, row, index_tuple, optional, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = op
        self.row = row
        self.index_tuple = index_tuple
        self.optional = optional
        self.pixmap = None
        if optional and not row.main_window.showing_structure():
            self.hide()

        font = self.font()
        font.setPointSize(int(font.pointSize() * 1.5))  # increase font size for better readability
        self.setFont(font)

        if self.op in {GraphExpression.LPAREN, GraphExpression.RPAREN}:
            self.pixmap = BRACKET_PIXMAPS[self.text()]
            self.setText("")
            self.setPixmap(self.pixmap)
            size_policy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.setSizePolicy(size_policy)

    def minimumSizeHint(self):
        if self.pixmap:
            return QSize(self.pixmap.width(), self.pixmap.height() // 2)
        return super().minimumSizeHint()

    def show_optional(self, show):
        if self.optional:
            if show:
                self.show()
            else:
                self.hide()

    def resizeEvent(self, event):
        if self.pixmap:
            size = self.size()
            scaled_pixmap = self.pixmap.scaled(size, Qt.IgnoreAspectRatio)
            self.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        context_menu.setStyleSheet(f"background-color: {default_menu_bg_color_name};")

        if self.op in {GraphExpression.SUM, GraphExpression.PROD}:
            flip_action = context_menu.addAction("Flip")
            flip_action.triggered.connect(self.on_flip)
            if self.index_tuple[-1] == 0:
                flip_action.setEnabled(False)  # can't flip at a leading minus or factor

            bracket_submenu = context_menu.addMenu("Add Brackets")
            if self.row.selecting_allowed():
                self.row.populate_bracket_menu(bracket_submenu, self.index_tuple)
            else:
                bracket_submenu.setEnabled(False)

            union_action = context_menu.addAction("Disjoint Union")
            union_action.triggered.connect(self.on_union)
            if not (self.row.selecting_allowed() and self.row.can_disjoint_union(self.index_tuple)):
                union_action.setEnabled(False)

        if self.op == GraphExpression.LPAREN and len(self.index_tuple) > 0:
            insert_neutral_submenu = context_menu.addMenu("Insert Neutral")
            self.row.populate_insert_neutral_menu(insert_neutral_submenu, self.index_tuple)

        debug_action = context_menu.addAction("DEBUG")
        debug_action.triggered.connect(self.on_debug)

        # Display the context menu
        context_menu.exec(event.globalPos())

    def on_flip(self):
        self.row.flip(self.index_tuple)

    def on_union(self):
        self.row.do_disjoint_union(self.index_tuple)

    def on_debug(self):
        print(f"DEBUG: {self.op=} {self.index_tuple=} {self.width()=} {self.height()=}")


class GraphWithPos:
    def __init__(self, graph=None, pos=None, from_dict=None):
        if from_dict is None:
            assert graph is not None
            # clean node names (needed, e.g., for grid graphs)
            self.G = nx.convert_node_labels_to_integers(graph, label_attribute='old_label')
            self.graph_hash = GraphHash(graph)
            if pos is None:
                self.pos = layout_with_scaffold(self.G)
            else:
                # translate pos dict to new node labels
                self.pos = {node: pos[self.G.nodes[node]['old_label']] for node in self.G.nodes}

            self.selected_nodes = set()
            self.normalize_pos()
        else:
            # we're loading from a file
            assert from_dict["class"] == "GraphWithPos"
            self.G = nx.node_link_graph(from_dict["graph"])
            self.graph_hash = GraphHash(self.G)
            self.pos = {int(node): (x, y) for node, (x, y) in from_dict["pos"].items()}
            self.selected_nodes = set(from_dict["selected_nodes"])

    def rehash(self):
        self.graph_hash = GraphHash(self.G)

    def select_all(self):
        self.selected_nodes = set(self.G.nodes)

    def deselect_all(self):
        self.selected_nodes = set()

    def normalize_pos(self):
        """
        Scale all `pos` coordinates so that the graph fits in a bounding rectangle of (-1,-1) ... (1, 1)
        (only if there are at least 2 nodes)
        """
        data = [self.pos[node] for node in self.G.nodes]
        if len(data) >= 2:
            data_x, data_y = zip(*data)
            min_x = min(data_x)
            max_x = max(data_x)
            min_y = min(data_y)
            max_y = max(data_y)
            delta_x = max_x-min_x
            if delta_x == 0:
                delta_x = 0.01
            delta_y = max_y-min_y
            if delta_y == 0:
                delta_y = 0.01
            self.pos = {node: ((self.pos[node][0]-min_x) / delta_x * 2-1,
                               (self.pos[node][1]-min_y) / delta_y * 2-1) for node in self.G.nodes}

    def to_latex_raw(self):
        node_options = {}
        node_labels = {}
        for node in self.G.nodes:
            node_options[node] = "selected" if node in self.selected_nodes else "unselected"
            node_labels[node] = ""

        # scale coordinates for LaTeX such that x and y ranges are [-1, 1]
        data = [self.pos[node] for node in self.G.nodes]
        if not data:
            # we're drawing the empty graph
            latex = r"""
    \begin{tikzpicture}[show background rectangle,scale=0.4, baseline={([yshift=-0.5ex]current bounding box.center)}]
        \node[invisible] at (-1,-1) {};
        \node[invisible] at (1,1) {};
    \end{tikzpicture}
"""
        elif len(data) == 1:
            latex = r"""
    \begin{tikzpicture}[show background rectangle,scale=0.4, baseline={([yshift=-0.5ex]current bounding box.center)}]
        \node[invisible] at (-1,-1) {};
        \node[invisible] at (1,1) {};
"""
            node = list(self.G.nodes)[0]
            sel = "selected" if node in self.selected_nodes else "unselected"
            latex += f"        \\node[{sel}] at (0,0) {{}};\n"
            latex += r"    \end{tikzpicture}" + "\n"
        else:
            tikz_options = "show background rectangle,scale=0.4,baseline={([yshift=-0.5ex]current bounding box.center)}"
            latex = nx.to_latex_raw(self.G, pos=deepcopy(self.pos), node_options=node_options,
                                    node_label=node_labels, tikz_options=tikz_options)
            # TODO report bug in NetworkX: it converts the pos entries to strings

        return r"\fbox{" + latex + "}"

    def selection_is_clique(self):
        nodes = list(self.selected_nodes)
        for i in range(1, len(nodes)):
            for j in range(i):
                if not self.G.has_edge(nodes[i], nodes[j]):
                    return False
        return True

    def to_dict(self):
        """
        :return: a dict for JSON serialisation
        """
        # we need to make a copy of our graph with just the nodes and edges
        # because nx.node_link_data() may otherwise return a dict that is not JSON serializable
        # TODO report this as a bug in NetworkX. contraction causes problems.
        plain_graph = nx.Graph()
        plain_graph.add_nodes_from(self.G.nodes)
        plain_graph.add_edges_from(self.G.edges)

        return {
            'class': 'GraphWithPos',
            'graph': nx.node_link_data(plain_graph),
            'pos': {node: [x, y] for node, (x, y) in self.pos.items()},
            'selected_nodes': list(self.selected_nodes)
        }

    def __repr__(self):
        return f"G<{self.graph_hash}>"

    def __eq__(self, other):
        """
        GraphWithPos objects are considered equal when their graphs are isomorphic.
        Instead of fully testing isomorphism, we just use equality of the graph hashes.
        :param other: the object to compare with
        :return: True if this graph is (most likely) isomorphic to the other one.
        """
        if isinstance(other, GraphWithPos):
            return self.graph_hash == other.graph_hash  # and nx.is_isomorphic(self.G, other.G) ?
        return False


class GraphExpression:
    SUM = 1  # this encodes the type and the precedence level
    PROD = 2
    # TODO the following isn't really clean design and should probably be refactored
    LPAREN = 10  # this is used as an operation type for an OpLabel only
    RPAREN = 11  # this is used as an operation type for an OpLabel only

    open_parens = {SUM: "(", PROD: "["}
    close_parens = {SUM: ")", PROD: "]"}

    def __init__(self, graph_w_pos_or_expr=None, item=None, items=None, op=SUM, from_dict=None):
        if from_dict is None:
            self.op = op
            if items is not None:
                self.items = items
            elif item is not None:
                self.items = [item]
            elif graph_w_pos_or_expr is None:
                self.items = []
            else:
                assert isinstance(graph_w_pos_or_expr, GraphWithPos) or isinstance(graph_w_pos_or_expr, GraphExpression)
                self.items = [(graph_w_pos_or_expr, 1)]
        else:
            # we're loading from a dict
            assert from_dict["class"] == "GraphExpression"
            self.op = from_dict["op"]
            self.items = []
            for item_dict, multiplicity in from_dict["items"]:
                if item_dict["class"] == "GraphExpression":
                    item = GraphExpression(from_dict=item_dict)
                else:
                    assert item_dict["class"] == "GraphWithPos"
                    item = GraphWithPos(from_dict=item_dict)
                self.items.append((item, multiplicity))

    def insert(self, graph_expr, multiplicity_delta, at_index=None, in_front=False):
        if at_index is None:
            for i in range(len(self.items)):
                graph_w_pos, multiplicity = self.items[i]
                if isinstance(graph_w_pos, GraphWithPos) and isinstance(graph_expr, GraphWithPos) and \
                        nx.is_isomorphic(graph_w_pos.G, graph_expr.G):
                    if multiplicity + multiplicity_delta == 0:
                        if self.op == self.PROD and len(self.items) == 1:
                            # can't delete last factor from a product. Replace it by the empty graph (1)
                            self.items[i] = GraphWithPos(nx.empty_graph(0, create_using=nx.Graph)), 1
                        else:
                            del self.items[i]
                    else:
                        self.items[i] = (graph_w_pos, multiplicity+multiplicity_delta)
                    return
        if multiplicity_delta != 0:
            if at_index is None:
                if in_front:
                    self.items.insert(0, (deepcopy(graph_expr), multiplicity_delta))
                else:
                    self.items.append((deepcopy(graph_expr), multiplicity_delta))
            else:
                self.items.insert(at_index, (deepcopy(graph_expr), multiplicity_delta))

    def create_widgets(self, row, index_tuple=()):
        outer_optional = index_tuple == () or self.op == self.PROD
        widgets = [OpLabel(self.LPAREN, row, index_tuple, outer_optional, self.open_parens[self.op])]

        first = True
        for i in range(len(self.items)):
            item, multiplicity = self.items[i]
            new_index_tuple = index_tuple + (i,)

            if self.op == self.SUM:
                if multiplicity == 1:
                    if not first:
                        widgets.append(OpLabel(GraphExpression.SUM, row, new_index_tuple, False, "+"))
                elif multiplicity == -1:
                    widgets.append(OpLabel(GraphExpression.SUM, row, new_index_tuple, False, "–"))
                else:
                    text = pretty_minus(f"{multiplicity}" if first else f"{multiplicity:+}")
                    widgets.append(OpLabel(GraphExpression.SUM, row, new_index_tuple, False, text))
            else:
                if not first:
                    widgets.append(OpLabel(GraphExpression.PROD, row, new_index_tuple, False, TIMES_SYMBOL))

            if isinstance(item, GraphWithPos):
                widgets.append(GraphWidget(graph_with_pos=item, row=row, index_tuple=new_index_tuple))
            else:
                assert isinstance(item, GraphExpression)
                widgets += item.create_widgets(row, new_index_tuple)

            if self.op == self.PROD and multiplicity != 1:
                exponent_label = QLabel(f"{pretty_minus(str(multiplicity))}")
                exponent_label.setAlignment(Qt.AlignTop)
                font = exponent_label.font()
                font.setPointSize(int(font.pointSize() * 1.5))  # increase font size for better readability
                exponent_label.setFont(font)
                widgets.append(exponent_label)
            first = False

        widgets.append(OpLabel(self.RPAREN, row, index_tuple, outer_optional, self.close_parens[self.op]))

        return widgets

    def index_tuple_lens(self, index_tuple, force_op=None) -> Tuple[GraphExpression, int]:
        """
        :param index_tuple: the index tuple to focus on
        :param force_op: if not None, modify expression so that it is of type `force_op` if it isn't already
        :return: a "lens" consisting of a GraphExpression and an index
        """
        if len(index_tuple) == 1:
            lens = self, index_tuple[0]
        else:
            sub_expr, _ = self.items[index_tuple[0]]
            lens = sub_expr.index_tuple_lens(index_tuple[1:])
        if force_op is None or force_op == lens[0].op:
            return lens
        sub_expr, i = lens
        expr, multiplicity = sub_expr.items[i]
        if len(sub_expr.items) == 1 and multiplicity == 1:
            # if there's only one item with multiplicity 1 we can simply change the operation type
            sub_expr.op = force_op
            return lens
        new_sub_expr = GraphExpression(item=(expr, 1), op=force_op)
        sub_expr.items[i] = (new_sub_expr, multiplicity)
        return new_sub_expr, 0

    def at_index_tuple(self, index_tuple):
        expr, i = self.index_tuple_lens(index_tuple)
        return expr.items[i]

    def deselect_all(self):
        for expr, _ in self.items:
            expr.deselect_all()

    def to_latex_raw(self):
        result = ""
        if self.op == self.SUM:
            first = True
            for expr, multiplicity in self.items:
                if multiplicity == 1:
                    if not first:
                        result += "+"
                elif multiplicity == -1:
                    result += "-"
                else:
                    result += f"{multiplicity}" if first else f"{multiplicity:+}"
                result += expr.to_latex_raw()
                first = False
        else:
            assert self.op == self.PROD

            def latex_helper(terms, first=True):
                result = ""
                for term, multiplicity in terms:
                    if not first:
                        result += f"{TIMES_SYMBOL_LATEX} "
                    if isinstance(term, GraphExpression) and term.op < self.op:
                        result += r"\left("
                    result += term.to_latex_raw()
                    if isinstance(term, GraphExpression) and term.op < self.op:
                        result += r"\right)"
                    if multiplicity != 1:
                        result += f"^{{{multiplicity}}}"
                    first = False
                return result

            numerator = []
            denominator = []
            sub_expressions = []
            for expr, multiplicity in self.items:
                if isinstance(expr, GraphExpression):
                    sub_expressions.append((expr, multiplicity))
                else:
                    assert isinstance(expr, GraphWithPos)
                    if multiplicity > 0:
                        numerator.append((expr, multiplicity))
                    elif multiplicity < 0:
                        denominator.append((expr, -multiplicity))

            if not numerator and not denominator:
                result = ""
                first = True
            elif not numerator:
                result = f"\\frac{{1}}{{{latex_helper(denominator)}}}"
                first = False
            elif not denominator:
                result = latex_helper(numerator)
                first = False
            else:
                result = f"\\frac{{{latex_helper(numerator)}}}{{{latex_helper(denominator)}}}"
                first = False

            result += latex_helper(sub_expressions, first=first)

        return result

    def simplify_nesting(self):
        # print(f"simplify_nesting({self=}")
        modified = False
        for i in range(len(self.items)-1, -1, -1):  # iterate backwards in case items get deleted
            sub_expr, multiplicity = self.items[i]
            if isinstance(sub_expr, GraphExpression):
                modified = sub_expr.simplify_nesting() or modified
                if not sub_expr.items:
                    modified = True
                    del self.items[i]
                elif sub_expr.op == self.op:
                    # expand sub-expression into our items list, taking multiplicity into account
                    modified = True
                    del self.items[i]
                    for sub2_expr, sub_multiplicity in sub_expr.items[::-1]:
                        self.items.insert(i, (sub2_expr, multiplicity * sub_multiplicity))
                elif len(sub_expr.items) == 1:
                    sub2_expr, sub_multiplicity = sub_expr.items[0]
                    if sub_multiplicity == 1:
                        modified = True
                        self.items[i] = sub2_expr, multiplicity

        if self.op == self.PROD:
            # remove empty graphs from products unless that would make the product empty
            empty_indices = []
            for i in range(len(self.items)):
                sub_expr, multiplicity = self.items[i]
                if isinstance(sub_expr, GraphWithPos) and sub_expr.G.number_of_nodes() == 0:
                    empty_indices.append(i)
            if len(empty_indices) < len(self.items):
                for i in empty_indices[::-1]:
                    modified = True
                    del self.items[i]
            elif len(self.items) > 1:
                # all factors are powers of the empty graph. Simplify to one empty graph to the power of 1
                modified = True
                self.items = [(self.items[0][0], 1)]

        if len(self.items) == 1:
            sub_expr, multiplicity = self.items[0]
            if multiplicity == 1 and isinstance(sub_expr, GraphExpression):
                modified = True
                self.op = sub_expr.op
                self.items = sub_expr.items

        # print(f"{modified=}, {self=}")
        return modified

    def graph_list(self):
        """
        :return: a flat list of all GraphWithPos objects in this expression
        """
        result = []
        for expr, _ in self.items:
            if isinstance(expr, GraphWithPos):
                result.append(expr)
            else:
                assert isinstance(expr, GraphExpression)
                result += expr.graph_list()
        return result

    def to_dict(self):
        """
        :return: a dict for JSON serialisation
        """
        return {
            'class': 'GraphExpression',
            'op': self.op,
            'items': [[expr.to_dict(), multiplicity] for expr, multiplicity in self.items]
        }

    def __repr__(self):
        t = "SUM" if self.op == self.SUM else "PROD"
        return f"{t}<{hex(id(self))[-4:]}>(" + \
               ', '.join(f'({expr}, {multiplicity})' for expr, multiplicity in self.items) + ")"


class Row:
    def __init__(self, main_window, parent_row, explanation, graph_expr: GraphExpression, latex_explanation=None):
        self.main_window = main_window
        self.row_index = -1
        self.parent_row = parent_row
        self.explanation = explanation
        if latex_explanation is None:
            self.latex_explanation = f"\\text{{{explanation}}}" if explanation is not None else ""
        else:
            self.latex_explanation = latex_explanation
        self.graph_expr = graph_expr
        self.reference_count = 0

        self.row_label = RowLabel(row=self)
        self.format_row_label()

        self.container = QWidget()

        self.layout = QHBoxLayout(self.container)
        self.layout.addWidget(self.row_label)

        for widget in self.graph_expr.create_widgets(self):
            self.layout.addWidget(widget)

        self.layout.addStretch(1)  # push widgets to the left in each row

        self.set_background_color()

    def set_row_index(self, row_index):
        self.row_index = row_index
        self.format_row_label()

    def format_row_label(self):
        label_text = f"({self.row_index+1})"
        if self.parent_row:
            label_text += f" = ({self.parent_row.row_index+1})"
        if self.explanation:
            label_text += f" [{self.explanation}]"
        self.row_label.setText(label_text)

    def graph_widgets(self):
        """
        All graph widgets in this row.
        :return: an iterator over all widgets of class GraphWidget in this row
        """
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, GraphWidget):
                yield widget

    def set_graph_size(self, size):
        for widget in self.graph_widgets():
            widget.setFixedSize(size, size)

    def selecting_allowed(self):
        return self.reference_count == 0

    def editing_allowed(self):
        return self.selecting_allowed() and self.parent_row is None

    def set_background_color(self):
        if self.editing_allowed():
            self.container.setStyleSheet(f"background-color: {ROW_BACKGROUND_LIGHT};")
        elif self.selecting_allowed():
            self.container.setStyleSheet(f"background-color: {ROW_BACKGROUND_MEDIUM};")
        else:
            self.container.setStyleSheet(f"background-color: {ROW_BACKGROUND_DARK};")

    def highlight_isomorphic(self, index_tuple):
        g, _ = self.graph_expr.at_index_tuple(index_tuple)
        assert isinstance(g, GraphWithPos)
        for widget in self.graph_widgets():
            widget.set_highlight(g == widget.graph_with_pos)

    def unhighlight_all(self):
        for widget in self.graph_widgets():
            widget.set_highlight(False)

    def append_derived_row(self, new_graph_expr, explanation, latex_explanation=None):
        new_row = Row(self.main_window, self, explanation, new_graph_expr, latex_explanation=latex_explanation)
        self.reference_count += 1
        self.main_window.add_row(new_row)
        self.unhighlight_all()
        self.set_background_color()

    def broadcast_layout(self, index_tuple):
        g, _ = self.graph_expr.at_index_tuple(index_tuple)
        assert isinstance(g, GraphWithPos)
        for widget in self.graph_widgets():
            if g is not widget.graph_with_pos and g == widget.graph_with_pos:
                isomorphism = nx.vf2pp_isomorphism(g.G, widget.graph_with_pos.G)
                for node1, node2 in isomorphism.items():
                    widget.graph_with_pos.pos[node2] = g.pos[node1]
                widget.draw_graph()

    def selected_vertices(self):
        """ return list of tuples (index_tuple, selected_nodes) for all graphs with selected vertices """
        result = []
        for widget in self.graph_widgets():
            if widget.graph_with_pos.selected_nodes:
                result.append((widget.index_tuple, widget.graph_with_pos.selected_nodes))
        return result

    def can_add_identify(self, index_tuple):
        graph_with_pos, multiplicity = self.graph_expr.at_index_tuple(index_tuple)
        if len(graph_with_pos.selected_nodes) == 2:
            u, v = graph_with_pos.selected_nodes
            if not graph_with_pos.G.has_edge(u, v):
                return True
        return False

    def do_add_identify(self, index_tuple):
        graph_with_pos, multiplicity = self.graph_expr.at_index_tuple(index_tuple)
        assert len(graph_with_pos.selected_nodes) == 2
        u, v = graph_with_pos.selected_nodes
        assert not graph_with_pos.G.has_edge(u, v)

        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple, force_op=GraphExpression.SUM)
        graph_w_pos, multiplicity = sub_expr.items[i]
        contracted_graph = GraphWithPos(nx.contracted_nodes(graph_w_pos.G, u, v, self_loops=False))
        graph_w_pos.G.add_edge(u, v)
        graph_w_pos.rehash()  # TODO refactor graph_w_pos so that all changes to the graphs are methods
        sub_expr.insert(contracted_graph, multiplicity, at_index=i+1)

        self.append_derived_row(new_graph_expr, "AI")
        self.deselect_all_except(index_tuple)

    def can_delete_contract(self, index_tuple):
        graph_with_pos, multiplicity = self.graph_expr.at_index_tuple(index_tuple)
        if len(graph_with_pos.selected_nodes) == 2:
            u, v = graph_with_pos.selected_nodes
            if graph_with_pos.G.has_edge(u, v):
                return True
        return False

    def do_delete_contract(self, index_tuple):
        graph_with_pos, multiplicity = self.graph_expr.at_index_tuple(index_tuple)
        assert len(graph_with_pos.selected_nodes) == 2
        u, v = graph_with_pos.selected_nodes
        assert graph_with_pos.G.has_edge(u, v)

        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple, force_op=GraphExpression.SUM)
        graph_w_pos, multiplicity = sub_expr.items[i]
        contracted_graph = GraphWithPos(nx.contracted_nodes(graph_w_pos.G, u, v, self_loops=False))
        graph_w_pos.G.remove_edge(u, v)
        graph_w_pos.rehash()  # TODO refactor graph_w_pos so that all changes to the graphs are methods
        sub_expr.insert(contracted_graph, -multiplicity, at_index=i+1)

        new_graph_expr.simplify_nesting()

        self.append_derived_row(new_graph_expr, "DC")
        self.deselect_all_except(index_tuple)

    def can_clique_sep(self, index_tuple):
        graph_w_pos, _ = self.graph_expr.at_index_tuple(index_tuple)
        return graph_w_pos.selection_is_clique()

    def do_clique_sep(self, index_tuple):
        graph_w_pos, _ = self.graph_expr.at_index_tuple(index_tuple)
        assert graph_w_pos.selection_is_clique()
        clique = graph_w_pos.selected_nodes

        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple, force_op=GraphExpression.PROD)
        graph_w_pos, multiplicity = sub_expr.items[i]

        H = graph_w_pos.G.copy()

        if len(clique) >= 1 and len(list(nx.connected_components(H))) > 1:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information)
            message_box.setText("Graph is not connected")
            message_box.setInformativeText("If a graph is not connected, it can only be separated by the empty clique,"
                                           " i.e. no vertices must be selected.")
            message_box.setWindowTitle("Separate by Clique")
            message_box.setStandardButtons(QMessageBox.Ok)
            message_box.exec_()
            return

        H.remove_nodes_from(clique)
        components = list(nx.connected_components(H))
        if len(components) <= 1:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information)
            message_box.setText("Clique is not separating")
            message_box.setInformativeText("After removing the selected clique, the graph must be split into at least "
                                           "two components.")
            message_box.setWindowTitle("Separate by Clique")
            message_box.setStandardButtons(QMessageBox.Ok)
            message_box.exec_()
            return

        if clique:
            new_clique_graph = GraphWithPos(graph_w_pos.G.subgraph(clique).copy(), pos=graph_w_pos.pos)
            new_clique_graph.normalize_pos()
            sub_expr.items[i] = (new_clique_graph, -multiplicity * (len(components) - 1))
        else:
            del sub_expr.items[i]  # the empty graph has chromatic polynomial 1 and can be deleted
        for component in components:
            subgraph_nodes = component.union(clique)
            subgraph_w_pos = GraphWithPos(graph_w_pos.G.subgraph(subgraph_nodes).copy(), pos=graph_w_pos.pos)
            sub_expr.insert(subgraph_w_pos, multiplicity, at_index=i)

        self.append_derived_row(new_graph_expr, "separate")
        self.deselect_all_except(index_tuple)

    def can_clique_join(self, index_tuple):
        sel = self.selected_vertices()
        if len(sel) != 2:
            return False
        (index_tuple1, selected1), (index_tuple2, selected2) = sel
        if index_tuple not in (index_tuple1, index_tuple2):
            return False
        if len(selected1) != len(selected2):
            return False
        expr1, i1 = self.graph_expr.index_tuple_lens(index_tuple1)
        expr2, i2 = self.graph_expr.index_tuple_lens(index_tuple2)
        if expr1 != expr2:
            return False
        graph_w_pos1, multiplicity1 = expr1.items[i1]
        graph_w_pos2, multiplicity2 = expr1.items[i2]
        if not (multiplicity1 == multiplicity2 and abs(multiplicity1) == 1):
            return False
        if not (graph_w_pos1.selection_is_clique() and graph_w_pos2.selection_is_clique()):
            return False
        return expr1.op == GraphExpression.PROD

    def do_clique_join(self, index_tuple):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()

        sel = self.selected_vertices()
        assert len(sel) == 2
        (index_tuple1, selected1), (index_tuple2, selected2) = sel
        assert index_tuple in (index_tuple1, index_tuple2)
        assert len(selected1) == len(selected2)
        expr1, i1 = new_graph_expr.index_tuple_lens(index_tuple1)
        expr2, i2 = new_graph_expr.index_tuple_lens(index_tuple2)
        assert expr1 == expr2
        if index_tuple == index_tuple2:
            i1, i2 = i2, i1
            selected1, selected2 = selected2, selected1
        graph_w_pos1, multiplicity1 = expr1.items[i1]
        graph_w_pos2, multiplicity2 = expr1.items[i2]
        assert multiplicity1 == multiplicity2 and abs(multiplicity1) == 1
        assert graph_w_pos1.selection_is_clique() and graph_w_pos2.selection_is_clique()
        assert expr1.op == GraphExpression.PROD

        # Glue graphs together at cliques

        g1 = graph_w_pos1.G
        g2 = graph_w_pos2.G
        # sort cliques by y coordinate to match them visually
        clique1 = sorted(selected1, key=lambda node: graph_w_pos1.pos[node][1])
        clique2 = sorted(selected2, key=lambda node: graph_w_pos2.pos[node][1])

        # create mapping to relabel g2 to be disjoint from g1
        max_g1_node = max(g1.nodes)
        mapping_g2 = {node: (node+max_g1_node+1) for node in g2.nodes}

        # Override the mapping for clique2 nodes to match clique1
        for node_c1, node_c2 in zip(clique1, clique2):
            mapping_g2[node_c2] = node_c1

        relabeled_g2 = nx.relabel_nodes(g2, mapping_g2)
        glued_graph = nx.compose(g1, relabeled_g2)

        expr1.items[i1] = GraphWithPos(glued_graph), multiplicity1
        del expr1.items[i2]
        expr1.insert(GraphWithPos(nx.complete_graph(len(clique1))), multiplicity1)

        new_graph_expr.simplify_nesting()

        self.append_derived_row(new_graph_expr, "glue")

    def can_disjoint_union(self, index_tuple):
        sub_expr, i = self.graph_expr.index_tuple_lens(index_tuple)
        try:
            assert sub_expr.op == GraphExpression.PROD
            assert i > 0
            graph_w_pos1, multiplicity1 = sub_expr.items[i-1]
            graph_w_pos2, multiplicity2 = sub_expr.items[i]
            assert multiplicity1 == multiplicity2
            assert isinstance(graph_w_pos1, GraphWithPos)
            assert isinstance(graph_w_pos2, GraphWithPos)
        except AssertionError:
            return False
        return True

    def do_disjoint_union(self, index_tuple):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()

        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)
        assert sub_expr.op == GraphExpression.PROD
        assert i > 0
        graph_w_pos1, multiplicity1 = sub_expr.items[i-1]
        graph_w_pos2, multiplicity2 = sub_expr.items[i]
        assert multiplicity1 == multiplicity2
        assert isinstance(graph_w_pos1, GraphWithPos)
        assert isinstance(graph_w_pos2, GraphWithPos)

        sub_expr.items[i-1] = GraphWithPos(nx.disjoint_union(graph_w_pos1.G, graph_w_pos2.G)), multiplicity1
        del sub_expr.items[i]

        new_graph_expr.simplify_nesting()

        self.append_derived_row(new_graph_expr, "union")
        self.select_subset([index_tuple, index_tuple[:-1]+(i-1,)])

    def can_whitney_flip(self, index_tuple):
        graph_w_pos, _ = self.graph_expr.at_index_tuple(index_tuple)
        return len(graph_w_pos.selected_nodes) == 2

    def do_whitney_flip(self, index_tuple):
        new_graph_expr = deepcopy(self.graph_expr)
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)
        graph_w_pos, multiplicity = sub_expr.items[i]
        split = graph_w_pos.selected_nodes
        assert len(split) == 2
        u, v = split
        new_graph_expr.deselect_all()

        H = graph_w_pos.G.copy()

        if len(list(nx.connected_components(H))) > 1:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information)
            message_box.setText("Graph is not connected")
            message_box.setInformativeText("This function is only implemented for connected graphs."
                                           "Consider separating components and gluing them back after the flip.")
            message_box.setWindowTitle("Whitney Flip")
            message_box.setStandardButtons(QMessageBox.Ok)
            message_box.exec_()
            return

        H.remove_nodes_from(split)
        components = list(nx.connected_components(H))
        if len(components) != 2:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information)
            message_box.setText("Vertices not separating")
            message_box.setInformativeText("After removing the selected vertices, the graph must be split into "
                                           "exactly two components.")
            message_box.setWindowTitle("Whitney Flip")
            message_box.setStandardButtons(QMessageBox.Ok)
            message_box.exec_()
            return

        components.sort(key=len, reverse=True)
        subgraphs = [graph_w_pos.G.subgraph(component.union(split)) for component in components]
        new_graph = nx.compose(subgraphs[0], nx.relabel_nodes(subgraphs[1], {u: v, v: u}, copy=True))
        sub_expr.items[i] = GraphWithPos(new_graph, graph_w_pos.pos), multiplicity

        self.append_derived_row(new_graph_expr, "Whitney")
        self.deselect_all_except(index_tuple)

    def deselect_all_except(self, index_tuple):
        for widget in self.graph_widgets():
            if widget.index_tuple != index_tuple:
                widget.deselect_all()

    def select_subset(self, index_tuple_container):
        for widget in self.graph_widgets():
            if widget.index_tuple in index_tuple_container:
                widget.select_all()
            else:
                widget.deselect_all()

    def merge_isomorphic(self, index_tuple):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)
        graph_w_pos, multiplicity = sub_expr.items[i]
        iso_indices = []
        for j in range(len(sub_expr.items)):
            graph_w_pos2, multiplicity2 = sub_expr.items[j]
            if j != i and isinstance(graph_w_pos2, GraphWithPos) and graph_w_pos == graph_w_pos2 and \
                    nx.is_isomorphic(graph_w_pos.G, graph_w_pos2.G):  # to be mathematically sure
                iso_indices.append(j)
                multiplicity += multiplicity2

        if not iso_indices:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information)
            message_box.setText("Nothing to simplify")
            message_box.setInformativeText("No isomorphic copies of the selected graph were found on the same nesting "
                                           "level within the same sum or product.")
            message_box.setWindowTitle("Collect Isomorphic")
            message_box.setStandardButtons(QMessageBox.Ok)
            message_box.exec_()
            return

        self.select_subset([index_tuple] + [index_tuple[:-1]+(j,) for j in iso_indices])

        if multiplicity != 0:
            sub_expr.items[i] = (graph_w_pos, multiplicity)
        else:
            # all copies of this graph cancel out
            # delete item i altogether by adding it to iso_indices
            iso_indices = sorted(iso_indices + [i])

        for j in iso_indices[::-1]:
            del sub_expr.items[j]

        new_graph_expr.simplify_nesting()

        self.append_derived_row(new_graph_expr, "collect")

    def can_separate(self, index_tuple):
        _, multiplicity = self.graph_expr.at_index_tuple(index_tuple)
        return abs(multiplicity) > 1

    def do_separate(self, index_tuple):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)
        graph_w_pos, multiplicity = sub_expr.items[i]
        sign = multiplicity // abs(multiplicity)
        assert abs(multiplicity) > 1
        sub_expr.items[i] = (graph_w_pos, multiplicity - sign)
        sub_expr.insert(deepcopy(graph_w_pos), sign, at_index=i+1)

        self.append_derived_row(new_graph_expr, "split term")
        self.select_subset([index_tuple])

    def add_brackets_single(self, index_tuple):
        """
        Add brackets around a single graph.
        :param index_tuple: the index tuple of the selected graph
        :return: None
        """
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)
        graph_w_pos, multiplicity = sub_expr.items[i]
        opposite_op = GraphExpression.SUM if sub_expr.op == GraphExpression.PROD else GraphExpression.PROD
        sub_expr.items[i] = GraphExpression(graph_w_pos, op=opposite_op), multiplicity

        if sub_expr.op == GraphExpression.SUM:
            self.main_window.show_structure()  # turn on full structure view so user sees all brackets

        self.append_derived_row(new_graph_expr, "brackets")
        self.select_subset([index_tuple])

    def populate_bracket_menu(self, brackets_menu, index_tuple):
        sub_expr, i = self.graph_expr.index_tuple_lens(index_tuple)
        assert i > 0
        begin_i = i - 1
        l = len(sub_expr.items)
        for end_i in range(i, l):
            text = "x" if begin_i > 0 else "(x"
            for j in range(1, l):
                text += " ● " if j == i else " ○ "
                text += "(x" if j == begin_i else "x"
                if j == end_i:
                    text += ")"
            sub_action = brackets_menu.addAction(text)
            sub_action.triggered.connect(lambda checked, it=index_tuple, ei=end_i: self.add_brackets_multiple(it, ei))

    def add_brackets_multiple(self, index_tuple, end_i):
        """
        Add brackets around a range of sub-expressions. Will add two levels of nesting so that the inner
        ``GraphExpression`` has the same ``op`` as the current one.
        :param index_tuple: The index tuple of the *second* sub-expression to be included in the brackets.
        :param end_i: The index of the last sub-expression to be included in the brackets.
        :return: None
        """
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)
        assert i > 0
        begin_i = i - 1
        opposite_op = GraphExpression.SUM if sub_expr.op == GraphExpression.PROD else GraphExpression.PROD
        bracketed_items = sub_expr.items[begin_i:end_i+1]
        sub_expr.items[begin_i:end_i+1] = [(GraphExpression(GraphExpression(items=bracketed_items, op=sub_expr.op),
                                                            op=opposite_op),
                                            1)]

        self.main_window.show_structure()  # turn on full structure view so user sees all brackets

        self.append_derived_row(new_graph_expr, "brackets")
        self.select_subset([index_tuple[:-1]+(j,) for j in range(begin_i, end_i+1)])

    def flip(self, index_tuple):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)

        assert i > 0
        sub_expr.items[i-1], sub_expr.items[i] = sub_expr.items[i], sub_expr.items[i-1]

        self.append_derived_row(new_graph_expr, "flip")
        self.deselect_all_except(())

    def populate_insert_neutral_menu(self, insert_neutral_menu, index_tuple):
        singles = self.main_window.single_graphs_from_rows()
        if singles and self.selecting_allowed():
            sub_expr, i = self.graph_expr.index_tuple_lens(index_tuple)
            for row_index, graph_w_pos in singles:
                if sub_expr.op == GraphExpression.SUM:
                    text = f"+ ({row_index+1}) – ({row_index+1})"
                else:
                    text = f"{TIMES_SYMBOL} ({row_index+1}) / ({row_index+1})"
                sub_action = insert_neutral_menu.addAction(text)
                sub_action.triggered.connect(lambda checked, g=graph_w_pos, it=index_tuple:
                                             self.insert_neutral(it, g))
        else:
            insert_neutral_menu.setEnabled(False)

    def insert_neutral(self, index_tuple, graph_w_pos: GraphWithPos):
        g1 = deepcopy(graph_w_pos)
        g1.deselect_all()
        g2 = deepcopy(g1)

        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)

        sub_expr.insert(g2, -1, at_index=i)
        sub_expr.insert(g1, 1, at_index=i)
        latex_explanation = "{} + G - G" if sub_expr.op == GraphExpression.SUM else f"{{}} {TIMES_SYMBOL_LATEX} G / G"
        self.append_derived_row(new_graph_expr, "insert", latex_explanation=latex_explanation)
        self.select_subset([index_tuple])

    def can_distribute_right(self, index_tuple):
        expr, i = self.graph_expr.index_tuple_lens(index_tuple)
        if expr.op != GraphExpression.PROD:
            return False
        term_expr, term_multiplicity = expr.items[i]
        for j in range(i+1, len(expr.items)):
            sum_expr, sum_multiplicity = expr.items[j]
            if isinstance(sum_expr, GraphExpression) and sum_expr.op == GraphExpression.SUM:
                return sum_multiplicity % term_multiplicity == 0
        return False

    def do_distribute_right(self, index_tuple):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()

        expr, i = new_graph_expr.index_tuple_lens(index_tuple)
        assert expr.op == GraphExpression.PROD
        term_expr, term_multiplicity = expr.items[i]
        for j in range(i+1, len(expr.items)):
            sum_expr, sum_multiplicity = expr.items[j]
            if isinstance(sum_expr, GraphExpression) and sum_expr.op == GraphExpression.SUM:
                assert sum_multiplicity % term_multiplicity == 0
                multiplicity_delta = sum_multiplicity // term_multiplicity
                for k in range(len(sum_expr.items)):
                    summand_expr, _ = sum_expr.index_tuple_lens((k,), force_op=GraphExpression.PROD)
                    summand_expr.insert(term_expr, multiplicity_delta, in_front=True)
                del expr.items[i]
                break

        new_graph_expr.simplify_nesting()

        self.append_derived_row(new_graph_expr, "distribute")
        self.select_subset([index_tuple])

    def can_factor_out(self, index_tuple):
        if len(index_tuple) < 2:
            return False
        expr, _ = self.graph_expr.index_tuple_lens(index_tuple)
        if expr.op != GraphExpression.PROD:
            return False
        super_expr, _ = self.graph_expr.index_tuple_lens(index_tuple[:-1])
        return super_expr.op == GraphExpression.SUM

    def do_factor_out(self, index_tuple):
        assert len(index_tuple) >= 2
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()

        expr, i = new_graph_expr.index_tuple_lens(index_tuple)
        assert expr.op == GraphExpression.PROD
        term_expr, term_multiplicity = expr.items[i]
        sum_expr, _ = new_graph_expr.index_tuple_lens(index_tuple[:-1])
        assert sum_expr.op == GraphExpression.SUM
        if len(index_tuple) == 2:
            assert sum_expr is new_graph_expr
            new_graph_expr = GraphExpression(new_graph_expr, op=GraphExpression.PROD)
            super2_expr = new_graph_expr
            j = 0
        else:
            super2_expr, j = new_graph_expr.index_tuple_lens(index_tuple[:-2])
        assert super2_expr.op == GraphExpression.PROD

        super2_expr.insert(term_expr, term_multiplicity, at_index=j)
        for k in range(len(sum_expr.items)):
            summand_expr, summand_multiplicity = sum_expr.items[k]
            if not isinstance(summand_expr, GraphExpression):
                assert isinstance(summand_expr, GraphWithPos)
                sum_expr.items[k] = GraphExpression(summand_expr, op=GraphExpression.PROD), summand_multiplicity
                summand_expr, summand_multiplicity = sum_expr.items[k]
            assert isinstance(summand_expr, GraphExpression) and summand_expr.op == GraphExpression.PROD
            summand_expr.insert(term_expr, -term_multiplicity, in_front=True)

        new_graph_expr.simplify_nesting()

        self.append_derived_row(new_graph_expr, "factor out")
        self.select_subset([index_tuple])

    def simplify_nesting(self):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        if new_graph_expr.simplify_nesting():
            self.append_derived_row(new_graph_expr, "simplify")
            self.select_subset([])
        else:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information)
            message_box.setText("Nothing to simplify")
            message_box.setInformativeText("Brackets in this expression are already simplified. "
                                           "Try 'Collect Isomorphic' to combine isomorphic graphs.")
            message_box.setWindowTitle("Simplify")
            message_box.setStandardButtons(QMessageBox.Ok)
            message_box.exec_()

    def copy_as_new_row(self):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        new_row = Row(self.main_window, self.parent_row, self.explanation, new_graph_expr, self.latex_explanation)
        if self.parent_row:
            self.parent_row.reference_count += 1
        self.main_window.add_row(new_row)
        self.unhighlight_all()

    def derivation_to_latex_raw(self, right=False, is_final=True):
        """
        Return the derivation of the current row in LaTeX, following back the chain of `parent_row`s until the
        start.
        :param right: True to put explanation on the right side of the equation
        :param is_final: True if this is the last equation in the derivation
        :return: If is_final == True, a string with the complete LaTeX derivation.
            If is_final == False, a pair (result, carry) where result is the result so far and carry is a string
            to be inserted in front of the next equals sign.
        """
        if not self.parent_row:
            result = r"""
% in document preamble:
% \usepackage{amsmath}
% \usepackage{tikz}
% \usetikzlibrary{backgrounds}

\tikzset{%
    unselected/.style={circle,fill=blue,minimum size=1.3mm,inner sep=0pt},
    selected/.style={circle,fill=red,minimum size=1.3mm,inner sep=0pt},
    invisible/.style={circle,minimum size=1.3mm,inner sep=0pt,opacity=0},
    background rectangle/.style={fill=cyan!15},
}
\setlength{\fboxsep}{0.3ex}
\setlength{\fboxrule}{0pt}
\begin{alignat*}{2}
"""
            carry = self.graph_expr.to_latex_raw() + "\n"
        else:
            result, carry = self.parent_row.derivation_to_latex_raw(right, is_final=False)
            result += r"% \end{alignat*}\begin{alignat*}{2} % uncomment to break here" + "\n"
            result += f"%%%%% ({self.row_index+1}):\n"
            if right:
                result += f"{carry} &= {self.graph_expr.to_latex_raw()} &&\\quad({self.latex_explanation})  \\\\\n"
            else:
                result += f"{self.latex_explanation}\\colon&& {carry} &= {self.graph_expr.to_latex_raw()}  \\\\\n"
            carry = ""
        if is_final:
            result += carry
            result += r"\end{alignat*}" + "\n"
            return result
        else:
            return result, carry

    def to_dict(self):
        """
        :return: a dict for JSON serialisation
        """
        return {
            'class': 'Row',
            'graph_expr': self.graph_expr.to_dict(),
            'explanation': self.explanation,
            'latex_explanation': self.latex_explanation,
            'row_index': self.row_index,
            'parent_row_index': self.parent_row.row_index if self.parent_row else -1
        }


class GraphWidget(QWidget):
    size = 200

    def __init__(self, graph_with_pos=None, row=None, index_tuple=None):
        super().__init__()
        self.setFixedSize(self.size, self.size)

        self.row = row
        self.index_tuple = index_tuple
        if graph_with_pos is None:
            self.graph_with_pos = GraphWithPos(nx.empty_graph(0, create_using=nx.Graph))
        else:
            self.graph_with_pos = graph_with_pos
        self.dragging = False  # Flag to check if we are dragging a vertex
        self.dragging_ydata = None
        self.dragging_xdata = None
        self.dragged_node = None     # Store the node being dragged
        self.drag_occurred = False   # Flag to check if a drag action took place
        self.highlight = False       # if True highlight widget with light yellow background

        # Create a matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # Draw the graph initially
        self.draw_graph()

        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

    def select_all(self):
        self.graph_with_pos.select_all()
        self.draw_graph()

    def deselect_all(self):
        self.graph_with_pos.deselect_all()
        self.draw_graph()

    def draw_graph(self):
        self.ax.clear()
        node_colors = [
            'red' if node in self.graph_with_pos.selected_nodes else 'blue' for node in self.graph_with_pos.G.nodes
        ]
        nx.draw(self.graph_with_pos.G, pos=self.graph_with_pos.pos,
                ax=self.ax, with_labels=False, node_color=node_colors, node_size=100)

        if self.highlight:
            self.fig.patch.set_facecolor('#ffffe8')  # Light yellow
        else:
            self.fig.patch.set_facecolor('white')

        self.canvas.draw()

    def set_highlight(self, highlight):
        self.highlight = highlight
        self.draw_graph()

    def enterEvent(self, event):
        if self.row.selecting_allowed():
            self.row.highlight_isomorphic(self.index_tuple)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.row.selecting_allowed():
            self.row.unhighlight_all()
        super().leaveEvent(event)

    def on_press(self, event):
        # Ensure that only a left-click initiates a drag
        if event.button != 1:
            return

        # Check if a node was clicked
        nodes = list(self.graph_with_pos.G.nodes)
        if nodes and event.xdata is not None and event.ydata is not None:
            data = [self.graph_with_pos.pos[node] for node in nodes]
            data_x, data_y = zip(*data)
            distances = np.sqrt((data_x - event.xdata)**2 + (data_y - event.ydata)**2)

            # If a node is close enough to the click, consider it selected
            if min(distances) < 0.1:  # adjust this threshold if necessary
                i = np.argmin(distances)
                self.dragging = True
                self.dragging_xdata = event.xdata
                self.dragging_ydata = event.ydata
                self.dragged_node = nodes[i]
                # print(f"on_press: {event.xdata=}, {event.ydata=}, {self.dragged_node=}")

    def on_motion(self, event):
        if self.dragging and self.dragged_node is not None and event.xdata is not None and event.ydata is not None:
            distance_dragged = math.sqrt((self.dragging_xdata-event.xdata)**2 + (self.dragging_ydata-event.ydata)**2)
            # print(f"on_motion: {event.xdata=}, {event.ydata=}, {distance_dragged=}, {self.dragged_node=}")
            if distance_dragged > 0.02:
                self.drag_occurred = True
            self.graph_with_pos.pos[self.dragged_node] = (event.xdata, event.ydata)
            self.draw_graph()

    def on_release(self, _):
        # If dragging didn't occur, toggle the node's state
        if not self.drag_occurred and self.dragged_node is not None and self.row.selecting_allowed():
            if self.dragged_node in self.graph_with_pos.selected_nodes:
                self.graph_with_pos.selected_nodes.remove(self.dragged_node)
            else:
                self.graph_with_pos.selected_nodes.add(self.dragged_node)

        self.dragging = False
        self.dragged_node = None
        self.drag_occurred = False  # Reset the drag_occurred flag

        self.graph_with_pos.normalize_pos()
        self.draw_graph()

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        context_menu.setStyleSheet(f"background-color: {default_menu_bg_color_name};")

        create_node_action = context_menu.addAction("Create Vertex")
        create_node_action.triggered.connect(lambda: self.option_create_node(event.pos()))
        if not self.row.editing_allowed():
            create_node_action.setEnabled(False)

        toggle_edge_action = context_menu.addAction("Toggle Edge")
        toggle_edge_action.triggered.connect(self.option_toggle_edge)
        if len(self.graph_with_pos.selected_nodes) != 2 or not self.row.editing_allowed():
            toggle_edge_action.setEnabled(False)

        delete_nodes_action = context_menu.addAction("Delete Vertices")
        delete_nodes_action.triggered.connect(self.option_delete_nodes)
        if not self.graph_with_pos.selected_nodes or not self.row.editing_allowed():
            delete_nodes_action.setEnabled(False)

        context_menu.addSeparator()

        spring_layout_action = context_menu.addAction("Spring Layout")
        spring_layout_action.triggered.connect(self.option_spring_layout)

        kamada_layout_action = context_menu.addAction("Kamada–Kawai Layout")
        kamada_layout_action.triggered.connect(self.option_kamada_kawai_layout)

        planar_layout_action = context_menu.addAction("Planar Layout")
        planar_layout_action.triggered.connect(self.option_planar_layout)

        broadcast_layout_action = context_menu.addAction("Broadcast Layout to Isomorphic")
        broadcast_layout_action.triggered.connect(self.option_broadcast_layout)

        max_clique_action = context_menu.addAction("Find Maximum Clique")
        max_clique_action.triggered.connect(self.option_max_clique)
        if not self.row.selecting_allowed():
            max_clique_action.setEnabled(False)

        context_menu.addSeparator()

        ai_action = context_menu.addAction("Addition-Identification")
        ai_action.triggered.connect(self.option_ai)
        if not self.row.can_add_identify(self.index_tuple) or not self.row.selecting_allowed():
            ai_action.setEnabled(False)

        dc_action = context_menu.addAction("Deletion-Contraction")
        dc_action.triggered.connect(self.option_dc)
        if not self.row.can_delete_contract(self.index_tuple) or not self.row.selecting_allowed():
            dc_action.setEnabled(False)

        clique_sep_action = context_menu.addAction("Separate at Clique")
        clique_sep_action.triggered.connect(self.option_clique_sep)
        if not self.row.can_clique_sep(self.index_tuple) or not self.row.selecting_allowed():
            clique_sep_action.setEnabled(False)

        clique_join_action = context_menu.addAction("Glue at Clique")
        clique_join_action.triggered.connect(self.option_clique_join)
        if not self.row.can_clique_join(self.index_tuple):  # this operation doesn't change selection, no check req'd
            clique_join_action.setEnabled(False)

        whitney_flip_action = context_menu.addAction("Whitney Flip")
        whitney_flip_action.triggered.connect(self.option_whitney_flip)
        if not self.row.selecting_allowed() or not self.row.can_whitney_flip(self.index_tuple):
            whitney_flip_action.setEnabled(False)

        context_menu.addSeparator()

        merge_isomorphic_action = context_menu.addAction("Collect Isomorphic")
        merge_isomorphic_action.triggered.connect(self.option_merge_isomorphic)
        if not self.row.selecting_allowed():
            merge_isomorphic_action.setEnabled(False)

        separate_action = context_menu.addAction("Split Term")
        separate_action.triggered.connect(self.option_separate)
        if not self.row.can_separate(self.index_tuple) or not self.row.selecting_allowed():
            separate_action.setEnabled(False)

        add_brackets_action = context_menu.addAction("Add Brackets")
        add_brackets_action.triggered.connect(self.option_add_brackets)
        if not self.row.selecting_allowed():
            add_brackets_action.setEnabled(False)

        insert_neutral_submenu = context_menu.addMenu("Insert Neutral")
        self.row.populate_insert_neutral_menu(insert_neutral_submenu, self.index_tuple)

        distribute_right_action = context_menu.addAction("Distribute Right")
        distribute_right_action.triggered.connect(self.option_distribute_right)
        if not self.row.can_distribute_right(self.index_tuple) or not self.row.selecting_allowed():
            distribute_right_action.setEnabled(False)

        factor_out_action = context_menu.addAction("Factor Out")
        factor_out_action.triggered.connect(self.option_factor_out)
        if not self.row.can_factor_out(self.index_tuple) or not self.row.selecting_allowed():
            factor_out_action.setEnabled(False)

        context_menu.addSeparator()

        copy_action = context_menu.addAction("Copy as new Row")
        copy_action.triggered.connect(self.option_copy)

        context_menu.addSeparator()

        test_action = context_menu.addAction("DEBUG")
        test_action.triggered.connect(self.option_test)

        # Show the context menu at the cursor's position
        context_menu.exec(event.globalPos())

    def option_create_node(self, position):
        # Adjust for canvas position within the widget
        canvas_pos = self.canvas.pos()
        adjusted_x = position.x()-canvas_pos.x()

        # Adjust the y-coordinate for the discrepancy between coordinate systems
        adjusted_y = self.canvas.height()-(position.y()-canvas_pos.y())

        # Convert adjusted widget coordinates (in pixels) to data coordinates
        inv_transform = self.ax.transData.inverted()
        data_pos = inv_transform.transform((adjusted_x, adjusted_y))

        # Create a new node at the click position
        new_node = max(self.graph_with_pos.G.nodes())+1 if self.graph_with_pos.G.nodes() else 0
        self.graph_with_pos.G.add_node(new_node)
        self.graph_with_pos.pos[new_node] = data_pos

        self.graph_with_pos.rehash()
        self.draw_graph()

    def option_toggle_edge(self):
        if len(self.graph_with_pos.selected_nodes) == 2:
            u, v = self.graph_with_pos.selected_nodes
            if self.graph_with_pos.G.has_edge(u, v):
                self.graph_with_pos.G.remove_edge(u, v)
            else:
                self.graph_with_pos.G.add_edge(u, v)
            self.graph_with_pos.rehash()
            self.draw_graph()  # Redraw the graph to reflect the changes

    def option_delete_nodes(self):
        for node in self.graph_with_pos.selected_nodes:
            self.graph_with_pos.G.remove_node(node)
            del self.graph_with_pos.pos[node]
        self.graph_with_pos.selected_nodes = set()
        self.graph_with_pos.rehash()
        self.draw_graph()

    def option_spring_layout(self):
        self.graph_with_pos.pos = layout_with_scaffold(self.graph_with_pos.G)
        self.draw_graph()

    def option_kamada_kawai_layout(self):
        self.graph_with_pos.pos = nx.kamada_kawai_layout(self.graph_with_pos.G)
        self.draw_graph()

    def option_planar_layout(self):
        try:
            self.graph_with_pos.pos = nx.planar_layout(self.graph_with_pos.G)
            self.draw_graph()
        except NetworkXException:
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information)
            message_box.setText("Graph not planar")
            message_box.setInformativeText("This graph appears not to be planar, so a planar layout is not possible.")
            message_box.setWindowTitle("Planar Layout")
            message_box.setStandardButtons(QMessageBox.Ok)
            message_box.exec_()

    def option_broadcast_layout(self):
        self.row.broadcast_layout(self.index_tuple)

    def option_max_clique(self):
        max_clique = max(nx.find_cliques(self.graph_with_pos.G), key=len)
        self.graph_with_pos.selected_nodes = set(max_clique)
        self.draw_graph()

    def option_ai(self):
        self.row.do_add_identify(self.index_tuple)

    def option_dc(self):
        self.row.do_delete_contract(self.index_tuple)

    def option_clique_sep(self):
        self.row.do_clique_sep(self.index_tuple)

    def option_clique_join(self):
        self.row.do_clique_join(self.index_tuple)

    def option_whitney_flip(self):
        self.row.do_whitney_flip(self.index_tuple)

    def option_merge_isomorphic(self):
        self.row.merge_isomorphic(self.index_tuple)

    def option_separate(self):
        self.row.do_separate(self.index_tuple)

    def option_copy(self):
        new_graph_w_pos = deepcopy(self.graph_with_pos)
        new_graph_w_pos.deselect_all()
        self.row.main_window.new_graph_row(graph_w_pos=new_graph_w_pos)

    def option_add_brackets(self):
        self.row.add_brackets_single(self.index_tuple)

    def option_distribute_right(self):
        self.row.do_distribute_right(self.index_tuple)

    def option_factor_out(self):
        self.row.do_factor_out(self.index_tuple)

    def option_test(self):
        print(self.index_tuple)
        print(list(nx.generate_edgelist(self.graph_with_pos.G, data=False)))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('ChromaCert')
        self.setGeometry(100, 100, 800, 600)

        self.view_structure_action = None
        self.create_menu()

        self.rows = []

        # Set up the scrollable canvas
        self.scroll_area = QScrollArea(self)
        self.setCentralWidget(self.scroll_area)
        self.scroll_area.setWidgetResizable(True)
        self.vertical_scroll_bar = self.scroll_area.verticalScrollBar()
        self.horizontal_scroll_bar = self.scroll_area.horizontalScrollBar()
        self.vertical_scroll_bar.rangeChanged.connect(self._scroll_to_bottom_left)
        self.prevent_auto_scroll = False

        # Container widget for the scroll area
        self.container = QWidget()
        self.container.setStyleSheet(f"background-color: {ROW_BACKGROUND_LIGHT};")
        self.scroll_area.setWidget(self.container)

        self.main_layout = QVBoxLayout(self.container)
        self.main_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)

        self.start_new_document()

    def start_new_document(self):
        clear_layout(self.main_layout)
        self.main_layout.addStretch()  # stretch always stays at the end to push widgets to the top
        self.rows = []

    def _scroll_to_bottom_left(self):
        if not self.prevent_auto_scroll:
            self.vertical_scroll_bar.setValue(self.vertical_scroll_bar.maximum())
            self.horizontal_scroll_bar.setValue(self.horizontal_scroll_bar.minimum())
        self.prevent_auto_scroll = False

    def create_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")
        file_new_action = file_menu.addAction("New")
        file_new_action.triggered.connect(self.on_file_new)

        file_open_action = file_menu.addAction("Open ...")
        file_open_action.triggered.connect(self.on_file_open)

        file_save_as_action = file_menu.addAction("Save as ...")
        file_save_as_action.triggered.connect(self.on_file_save_as)

        new_menu = menu_bar.addMenu("New Graph")
        new_action_empty = new_menu.addAction("Empty")
        new_action_empty.triggered.connect(lambda: self.new_graph_row(nx.empty_graph(0, create_using=nx.Graph)))

        new_submenu_complete = new_menu.addMenu("Complete")
        for i in range(2, 11):
            new_action_complete = new_submenu_complete.addAction(f"K_{i}")
            new_action_complete.triggered.connect(lambda checked, ii=i: self.new_graph_row(nx.complete_graph(ii)))

        new_submenu_cycle = new_menu.addMenu("Cycle")
        for i in range(3, 11):
            new_action_cycle = new_submenu_cycle.addAction(f"C_{i}")
            new_action_cycle.triggered.connect(lambda checked, ii=i: self.new_graph_row(nx.cycle_graph(ii)))

        new_submenu_wheel = new_menu.addMenu("Wheel")
        for i in itertools.chain(range(3, 7), range(7, 20, 2)):
            new_action_wheel = new_submenu_wheel.addAction(f"W_{i}")
            new_action_wheel.triggered.connect(lambda checked, ii=i: self.new_graph_row(nx.wheel_graph(ii+1)))

        new_submenu_star = new_menu.addMenu("Star")
        for i in itertools.chain(range(3, 7), range(7, 20, 2)):
            new_action_star = new_submenu_star.addAction(f"St_{i}")
            new_action_star.triggered.connect(lambda checked, ii=i: self.new_graph_row(nx.star_graph(ii)))

        new_submenu_bipartite = new_menu.addMenu("Bipartite")
        for i in range(2, 6):
            for j in range(2, i+1):
                new_action_bipartite = new_submenu_bipartite.addAction(f"K_{{{i}, {j}}}")
                new_action_bipartite.triggered.connect(lambda checked, ii=i, jj=j:
                                                       self.new_graph_row(nx.complete_multipartite_graph(ii, jj)))

        new_submenu_grid = new_menu.addMenu("Grid")
        for i in range(3, 7):
            for j in range(2, i+1):
                new_action_grid = new_submenu_grid.addAction(f"{i} × {j} grid")
                new_action_grid.triggered.connect(lambda checked, ii=i, jj=j:
                                                  self.new_graph_row(nx.grid_2d_graph(ii, jj)))

        view_menu = menu_bar.addMenu("View")
        size_action_group = QActionGroup(self)
        sizes = [("Small", 150), ("Normal", 200), ("Large", 260), ("Extra Large", 350), ("Even Larger", 500)]

        for text, size in sizes:
            action = view_menu.addAction(f"{text} Graphs")
            action.setCheckable(True)
            action.triggered.connect(lambda checked, s=size: self.set_graph_size(s))
            size_action_group.addAction(action)

        default_size_i = 1  # Medium is the default
        size_action_group.actions()[1].setChecked(True)
        GraphWidget.size = sizes[default_size_i][1]

        view_menu.addSeparator()

        self.view_structure_action = view_menu.addAction("Full Expression Structure")
        self.view_structure_action.setCheckable(True)
        self.view_structure_action.setChecked(False)
        self.view_structure_action.triggered.connect(self.on_toggle_structure)

    def add_row(self, row: Row):
        row.set_row_index(len(self.rows))
        self.main_layout.insertWidget(self.main_layout.count()-1, row.container)  # insert before the final stretch
        self.rows.append(row)

    def new_graph_row(self, graph=None, graph_w_pos=None):
        if graph_w_pos is not None:
            row = Row(self, None, None, GraphExpression(graph_w_pos))
        else:
            row = Row(self, None, None, GraphExpression(GraphWithPos(graph)))
        self.add_row(row)

    def set_graph_size(self, size):
        GraphWidget.size = size
        for row in self.rows:
            row.set_graph_size(size)
        self.prevent_auto_scroll = True

    def on_toggle_structure(self, checked):
        for row in self.rows:
            for i in range(row.layout.count()):
                item = row.layout.itemAt(i)
                widget = item.widget()
                if widget and isinstance(widget, OpLabel):
                    widget.show_optional(checked)

    def showing_structure(self):
        return self.view_structure_action.isChecked()

    def show_structure(self):
        if not self.showing_structure():
            self.view_structure_action.setChecked(True)

    def single_graphs_from_rows(self):
        """
        Find all rows containing a single graph.
        :return: a list of tuples (row_index, graph_w_pos)
        """
        result = []
        for row in self.rows:
            l = row.graph_expr.graph_list()
            if len(l) == 1:
                result.append((row.row_index, l[0]))
        return result

    def on_file_new(self):
        self.start_new_document()

    def on_file_open(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open ChromaCert File", "",
                                                   "ChromaCert Files (*.chroma);;All Files (*)", options=options)
        if file_name:
            try:
                with open(file_name) as f:
                    file_dict = json.load(f)
                assert file_dict.get("class") == "ChromaCert" and file_dict.get("version") == 1 and \
                       "rows" in file_dict, "Invalid file format"
                self.start_new_document()
                for row_dict in file_dict["rows"]:
                    assert row_dict.get("class") == "Row"
                    graph_expr = GraphExpression(from_dict=row_dict["graph_expr"])
                    explanation = row_dict["explanation"]
                    assert row_dict["row_index"] == len(self.rows)
                    parent_row_index = row_dict["parent_row_index"]
                    parent_row = self.rows[parent_row_index] if parent_row_index >= 0 else None
                    if parent_row is not None:
                        assert isinstance(parent_row, Row)
                        parent_row.reference_count += 1
                    latex_explanation = row_dict["latex_explanation"]
                    row = Row(self, parent_row, explanation, graph_expr, latex_explanation)
                    self.add_row(row)

                QApplication.processEvents()  # without this, setting the background colors didn't fully work

                for row in self.rows:
                    row.set_background_color()

            except Exception as e:
                tb = traceback.format_exc()  # Get the full traceback as a string
                error_dialog = QMessageBox(self)  # Parent to MainWindow
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setText("Error")
                error_dialog.setInformativeText(str(e)+"\n\n"+tb)
                error_dialog.setWindowTitle("Error")
                error_dialog.exec_()  # This makes it modal

    def on_file_save_as(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save As", "", "ChromaCert Files (*.chroma);;All Files (*)",
                                                   options=options)
        if file_name:
            # If the user doesn't add the extension, add it for them
            if not file_name.lower().endswith('.chroma'):
                file_name += '.chroma'
            with open(file_name, "w") as f:
                json.dump(self.to_dict(), f)

    def to_dict(self):
        return {
            'class': 'ChromaCert',
            'version': 1,
            'rows': [row.to_dict() for row in self.rows]
        }


if __name__ == '__main__':
    app = QApplication(sys.argv)

    BRACKET_PIXMAPS = {
        "(": QPixmap("resources/lparen200.png"),
        ")": QPixmap("resources/rparen200.png"),
        "[": QPixmap("resources/lbracket200.png"),
        "]": QPixmap("resources/rbracket200.png"),
    }

    default_menu_palette = QMenu().palette()
    default_menu_bg_color_name = default_menu_palette.color(QPalette.Background).name()
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
