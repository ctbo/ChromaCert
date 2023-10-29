# ChromaCert (c) 2023 by Harald Bögeholz

from __future__ import annotations
from typing import Tuple

import sys
from copy import deepcopy
import math

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QMenu

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)

# configure multiplication symbol according to taste
# TIMES_SYMBOL = "·"
TIMES_SYMBOL_LATEX = r"\cdot"
TIMES_SYMBOL = "×"
# TIMES_SYMBOL_LATEX = r"\times"


class RowLabel(QLabel):
    def __init__(self, row=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row = row
        font = self.font()
        font.setBold(True)
        self.setFont(font)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        copy_action = context_menu.addAction("Copy as new Row")
        copy_action.triggered.connect(self.on_copy)

        latex_action = context_menu.addAction("Export as LaTeX")
        latex_action.triggered.connect(self.on_latex)

        debug_action = context_menu.addAction("DEBUG Row")
        debug_action.triggered.connect(self.on_debug)

        # Display the context menu
        context_menu.exec(event.globalPos())

    def on_copy(self):
        self.row.copy_as_new_row()

    def on_latex(self):
        print(self.row.derivation_to_latex_raw())

    def on_debug(self):
        print(self.row.graph_expr)


class OpLabel(QLabel):
    def __init__(self, op, row, index_tuple, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op = op
        self.row = row
        self.index_tuple = index_tuple

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        if self.op in {GraphExpression.SUM, GraphExpression.PROD}:
            flip_action = context_menu.addAction("Flip")
            flip_action.triggered.connect(self.on_flip)
            if self.index_tuple[-1] == 0:
                flip_action.setEnabled(False)  # can't flip at a leading minus or factor
        else:
            debug_action = context_menu.addAction("DEBUG")
            debug_action.triggered.connect(self.on_debug)

        # Display the context menu
        context_menu.exec(event.globalPos())

    def on_flip(self):
        self.row.flip(self.index_tuple)

    def on_debug(self):
        print(f"DEBUG: {self.op=} {self.index_tuple=}")


class GraphWithPos:
    def __init__(self, graph, pos=None):
        self.G = graph
        if pos is None:
            self.pos = nx.spring_layout(self.G)
        else:
            self.pos = pos.copy()
        self.selected_nodes = set()

    def select_all(self):
        self.selected_nodes = set(self.G.nodes)

    def deselect_all(self):
        self.selected_nodes = set()

    def to_latex_raw(self):
        node_options = {}
        node_labels = {}
        for node in self.G.nodes:
            node_options[node] = "selected" if node in self.selected_nodes else "unselected"
            node_labels[node] = ""
        latex = nx.to_latex_raw(self.G, pos=deepcopy(self.pos), node_options=node_options,
            node_label=node_labels,
            tikz_options="show background rectangle,scale=0.8, baseline={([yshift=-0.5ex]current bounding box.center)}")
        # TODO report bug in NetworkX. deepcopy() shouldn't be necessary, positions shouldn't be converted to strings
        return r"\fbox{" + latex + "}"

    def selection_is_clique(self):
        nodes = list(self.selected_nodes)
        for i in range(1, len(nodes)):
            for j in range(i):
                if not self.G.has_edge(nodes[i], nodes[j]):
                    return False
        return True

    def __str__(self):
        return "<G>"  # TODO this could show more information


class GraphExpression:
    SUM = 1    # this encodes the type and the precedence level
    PROD = 2
    # TODO the following isn't really clean design and should probably be refactored
    LPAREN = 10  # this is used as an operation type for an OpLabel only
    RPAREN = 11  # this is used as an operation type for an OpLabel only

    def __init__(self, graph_w_pos=None, item=None, op=SUM):
        self.op = op
        if item is not None:
            self.items = [item]
        elif graph_w_pos is None:
            self.items = []
        else:
            assert isinstance(graph_w_pos, GraphWithPos)
            self.items = [(graph_w_pos, 1)]

    def insert(self, graph_expr, multiplicity_delta, at_index=None):
        if at_index is None:
            for i in range(len(self.items)):
                item, multiplicity = self.items[i]
                if isinstance(item, GraphWithPos) and isinstance(graph_expr, GraphWithPos) and \
                        nx.is_isomorphic(item.G, graph_expr.G):
                    if multiplicity + multiplicity_delta == 0:
                        del self.items[i]
                    else:
                        self.items[i] = (item, multiplicity+multiplicity_delta)
                    return
        if multiplicity_delta != 0:
            if at_index is None:
                self.items.append((graph_expr, multiplicity_delta))
            else:
                self.items.insert(at_index, (graph_expr, multiplicity_delta))

    def create_widgets(self, row, index_tuple=()):
        if not self.items:
            return [QLabel(f"EMPTY {'SUM' if self.op == self.SUM else 'PROD'}")]
        widgets = []
        first = True
        for i in range(len(self.items)):
            item, multiplicity = self.items[i]
            new_index_tuple = index_tuple + (i,)

            if self.op == self.SUM:
                if multiplicity == 1:
                    if not first:
                        widgets.append(OpLabel(GraphExpression.SUM, row, new_index_tuple, "+"))
                elif multiplicity == -1:
                    widgets.append(OpLabel(GraphExpression.SUM, row, new_index_tuple, "-"))
                else:
                    widgets.append(OpLabel(GraphExpression.SUM, row, new_index_tuple,
                                           f"{multiplicity}" if first else f"{multiplicity:+}"))
            else:
                if not first:
                    widgets.append(OpLabel(GraphExpression.PROD, row, new_index_tuple, TIMES_SYMBOL))

            if isinstance(item, GraphWithPos):
                widgets.append(GraphWidget(graph_with_pos=item, row=row, index_tuple=new_index_tuple))
            else:
                assert isinstance(item, GraphExpression)
                if item.op < self.op:
                    widgets.append(OpLabel(GraphExpression.LPAREN, row, new_index_tuple, "("))
                widgets += item.create_widgets(row, new_index_tuple)
                if item.op < self.op:
                    widgets.append(OpLabel(GraphExpression.RPAREN, row, new_index_tuple, ")"))

            if self.op == self.PROD and multiplicity != 1:
                widgets.append(QLabel(f"^{multiplicity}"))
            first = False
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
        return (new_sub_expr, 0)

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

            def latex_helper(terms):
                result = ""
                first = True
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
            for expr, multiplicity in self.items:
                if multiplicity > 0:
                    numerator.append((expr, multiplicity))
                elif multiplicity < 0:
                    denominator.append((expr, -multiplicity))
            if not numerator:
                result = f"\\frac{{1}}{{{latex_helper(denominator)}}}"
            elif not denominator:
                result = latex_helper(numerator)
            else:
                result = f"\\frac{{{latex_helper(numerator)}}}{{{latex_helper(denominator)}}}"

        return result

    def simplify_nesting(self, parent=None, parent_i=None):
        for i in range(len(self.items)-1, -1, -1):  # iterate backwards in case items get deleted
            sub_expr, _ = self.items[i]
            if isinstance(sub_expr, GraphExpression):
                sub_expr.simplify_nesting(self, i)
        if len(self.items) > 1 or parent is None:
            return
        if not self.items:
            del parent.items[parent_i]
        else:
            expr, multiplicity = self.items[0]
            if multiplicity == 1:
                parent.items[parent_i] = expr, multiplicity

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

    def __str__(self):
        t = "SUM" if self.op == self.SUM else "PROD"
        return f"{t}({', '.join('('+str(expr)+', ' + str(multiplicity)+')' for expr, multiplicity in self.items)})"

class Row:
    def __init__(self, main_window, parent_row, explanation, graph_expr: GraphExpression, latex_explanation=None):
        self.main_window = main_window
        self.row_index = -1
        self.parent_row = parent_row
        self.explanation = explanation
        if latex_explanation is None:
            self.latex_explanation = f"\\text{{{explanation}}}"
        else:
            self.latex_explanation = latex_explanation
        self.graph_expr = graph_expr
        self.reference_count = 0  # TODO rows that are referenced by other rows shouldn't be edited

        self.row_label = RowLabel(row=self)
        self.format_row_label()

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.row_label)

        for widget in self.graph_expr.create_widgets(self):
            self.layout.addWidget(widget)

        self.layout.addStretch(1)  # push widgets to the left in each row

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

    def selecting_allowed(self):
        return self.reference_count == 0

    def editing_allowed(self):
        return self.selecting_allowed() and self.parent_row is None

    def selected_vertices(self):
        """ return list of tuples (index_tuple, selected_nodes) for all graphs with selected vertices """
        result = []
        for i in range(self.layout.count()):
            item = self.layout.itemAt(i)
            widget = item.widget()
            if widget and isinstance(widget, GraphWidget):
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
        sub_expr.insert(contracted_graph, multiplicity, at_index=i+1)

        new_row = Row(self.main_window, self, "AI", new_graph_expr)
        self.reference_count += 1
        self.main_window.add_row(new_row)
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
        sub_expr.insert(contracted_graph, -multiplicity, at_index=i+1)

        new_row = Row(self.main_window, self, "DC", new_graph_expr)
        self.reference_count += 1
        self.main_window.add_row(new_row)
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
        if len(graph_w_pos.selected_nodes) >= 1 and len(nx.connected_components(H)) > 1:
            # TODO modal dialog
            print("Can't separate by this clique: graph is not connected.")
            return
        H.remove_nodes_from(clique)
        components = list(nx.connected_components(H))
        if len(components) <= 1:
            # TODO modal dialog
            print("Clique isn't separating.")
            return
        if clique:
            sub_expr.items[i] = (GraphWithPos(graph_w_pos.G.subgraph(clique).copy(), pos=graph_w_pos.pos),
                                 -multiplicity * (len(components) - 1))
        else:
            del sub_expr.items[i]  # the empty graph has chromatic polynomial 1 and can be deleted
        for component in components:
            subgraph_nodes = component.union(clique)
            subgraph_w_pos = GraphWithPos(graph_w_pos.G.subgraph(subgraph_nodes).copy(), pos=graph_w_pos.pos)
            sub_expr.insert(subgraph_w_pos, multiplicity, at_index=i)

        new_row = Row(self.main_window, self, "separate", new_graph_expr)
        self.reference_count += 1
        self.main_window.add_row(new_row)
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

        new_row = Row(self.main_window, self, "glue", new_graph_expr)
        self.reference_count += 1
        self.main_window.add_row(new_row)

    def select_single_graph(self, index_tuple):
        for j in range(self.layout.count()):
            item = self.layout.itemAt(j)
            widget = item.widget()
            if widget and isinstance(widget, GraphWidget):
                if widget.index_tuple == index_tuple:
                    widget.select_all()
                else:
                    widget.deselect_all()

    def deselect_all_except(self, index_tuple):
        for j in range(self.layout.count()):
            item = self.layout.itemAt(j)
            widget = item.widget()
            if widget and isinstance(widget, GraphWidget):
                if widget.index_tuple != index_tuple:
                    widget.deselect_all()

    def merge_isomorphic(self, index_tuple):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)
        graph_w_pos, multiplicity = sub_expr.items[i]
        iso_indices = []
        for j in range(len(sub_expr.items)):
            graph_w_pos2, multiplicity2 = sub_expr.items[j]
            if j != i and isinstance(graph_w_pos2, GraphWithPos) and nx.is_isomorphic(graph_w_pos.G, graph_w_pos2.G):
                iso_indices.append(j)
                multiplicity += multiplicity2

        if not iso_indices:
            # TODO modal dialog?
            print("Nothing to simplify.")
            return

        if multiplicity != 0:
            sub_expr.items[i] = (graph_w_pos, multiplicity)
        else:
            # all copies of this graph cancel out
            # delete item i altogether by adding it to iso_indices
            iso_indices = sorted(iso_indices + [i])

        for j in iso_indices[::-1]:
            del sub_expr.items[j]

        new_graph_expr.simplify_nesting()

        new_row = Row(self.main_window, self, "collect", new_graph_expr)
        self.reference_count += 1
        self.main_window.add_row(new_row)
        self.select_single_graph(index_tuple)

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

        new_row = Row(self.main_window, self, "split term", new_graph_expr)
        self.reference_count += 1
        self.main_window.add_row(new_row)
        self.select_single_graph(index_tuple)

    def flip(self, index_tuple):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)

        assert i > 0
        sub_expr.items[i-1], sub_expr.items[i] = sub_expr.items[i], sub_expr.items[i-1]

        new_row = Row(self.main_window, self, "flip", new_graph_expr)
        self.reference_count += 1
        self.main_window.add_row(new_row)
        self.deselect_all_except(())

    def insert_neutral(self, index_tuple, graph_w_pos: GraphWithPos):
        g1 = deepcopy(graph_w_pos)
        g1.deselect_all()
        g2 = deepcopy(g1)

        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        sub_expr, i = new_graph_expr.index_tuple_lens(index_tuple)

        sub_expr.insert(g2, -1, at_index=i)
        sub_expr.insert(g1, 1, at_index=i)

        new_row = Row(self.main_window, self, "insert", new_graph_expr)
        self.reference_count += 1
        self.main_window.add_row(new_row)
        self.select_single_graph(index_tuple)

    def copy_as_new_row(self):
        new_graph_expr = deepcopy(self.graph_expr)
        new_graph_expr.deselect_all()
        new_row = Row(self.main_window, self.parent_row, self.explanation, new_graph_expr, self.latex_explanation)
        if self.parent_row:
            self.parent_row.reference_count += 1
        if __name__ == '__main__':
            self.main_window.add_row(new_row)

    def derivation_to_latex_raw(self, is_final=True):
        if not self.parent_row:
            result = r"\setlength{\fboxsep}{0.3ex}" + "\n"
            result += r"\setlength{\fboxrule}{0pt}" + "\n"
            result += r"\[\begin{array}{r@{\quad}c@{\quad}l}" + "\n"
            result += self.graph_expr.to_latex_raw() + "\n"
        else:
            result = self.parent_row.derivation_to_latex_raw(is_final=False)
            result += f"& \\stackrel{{{self.latex_explanation}}}{{=}} &\n {self.graph_expr.to_latex_raw()} \\\\\n"
        if is_final:
            result += "\\end{array}\\]\n"
        return result


class GraphWidget(QWidget):
    def __init__(self, graph_with_pos=None, row=None, index_tuple=None):
        super().__init__()
        self.setFixedSize(200, 200)  # Set a fixed size for each graph widget

        self.row = row
        self.index_tuple = index_tuple
        if graph_with_pos is None:
            self.graph_with_pos = GraphWithPos(nx.empty_graph(0, create_using=nx.Graph))
        else:
            self.graph_with_pos = graph_with_pos
        self.dragging = False  # Flag to check if we are dragging a vertex
        self.dragging_ydata = None
        self.dragging_xdata = None
        self.dragged_node = None  # Store the node being dragged
        self.drag_occurred = False  # Flag to check if a drag action took place

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
            'red' if node in self.graph_with_pos.selected_nodes else 'blue' for node in self.graph_with_pos.G.nodes()
        ]
        nx.draw(self.graph_with_pos.G, pos=self.graph_with_pos.pos,
                ax=self.ax, with_labels=False, node_color=node_colors, node_size=100)
        self.canvas.draw()

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
            self.draw_graph()

        self.dragging = False
        self.dragged_node = None
        self.drag_occurred = False  # Reset the drag_occurred flag

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

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

        spring_layout_action = context_menu.addAction("Spring Layout")
        spring_layout_action.triggered.connect(self.option_spring_layout)

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

        context_menu.addSeparator()

        merge_isomorphic_action = context_menu.addAction("Collect Isomorphic")
        merge_isomorphic_action.triggered.connect(self.option_merge_isomorphic)
        if not self.row.selecting_allowed():
            merge_isomorphic_action.setEnabled(False)

        separate_action = context_menu.addAction("Split Term")
        separate_action.triggered.connect(self.option_separate)
        if not self.row.can_separate(self.index_tuple) or not self.row.selecting_allowed():
            separate_action.setEnabled(False)

        copy_action = context_menu.addAction("Copy as new Row")
        copy_action.triggered.connect(self.option_copy)

        insert_neutral_submenu = context_menu.addMenu("Insert Neutral")
        singles = self.row.main_window.single_graphs_from_rows()
        if singles and self.row.selecting_allowed():
            sub_expr, i = self.row.graph_expr.index_tuple_lens(self.index_tuple)
            for row_index, graph_w_pos in singles:
                if sub_expr.op == GraphExpression.SUM:
                    text = f"+ ({row_index+1}) – ({row_index+1})"
                else:
                    text = f"{TIMES_SYMBOL} ({row_index+1}) / ({row_index+1})"
                sub_action = insert_neutral_submenu.addAction(text)
                sub_action.triggered.connect(lambda checked, g=graph_w_pos: self.option_insert_neutral(g))
        else:
            insert_neutral_submenu.setEnabled(False)

        test_action = context_menu.addAction("Test")
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

        # Redraw the graph
        self.draw_graph()

    def option_toggle_edge(self):
        if len(self.graph_with_pos.selected_nodes) == 2:
            u, v = self.graph_with_pos.selected_nodes
            if self.graph_with_pos.G.has_edge(u, v):
                self.graph_with_pos.G.remove_edge(u, v)
            else:
                self.graph_with_pos.G.add_edge(u, v)
            self.draw_graph()  # Redraw the graph to reflect the changes

    def option_delete_nodes(self):
        for node in self.graph_with_pos.selected_nodes:
            self.graph_with_pos.G.remove_node(node)
            del self.graph_with_pos.pos[node]
        self.graph_with_pos.selected_nodes = set()
        self.draw_graph()

    def option_spring_layout(self):
        self.graph_with_pos.pos = nx.spring_layout(self.graph_with_pos.G)
        self.draw_graph()

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

    def option_merge_isomorphic(self):
        self.row.merge_isomorphic(self.index_tuple)

    def option_separate(self):
        self.row.do_separate(self.index_tuple)

    def option_copy(self):
        new_graph_w_pos = deepcopy(self.graph_with_pos)
        new_graph_w_pos.deselect_all()
        self.row.main_window.new_graph_row(graph_w_pos=new_graph_w_pos)

    def option_insert_neutral(self, graph_w_pos):
        self.row.insert_neutral(self.index_tuple, graph_w_pos)

    def option_test(self):
        print(nx.chromatic_polynomial(self.graph_with_pos.G))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('ChromaCert')
        self.setGeometry(100, 100, 800, 600)

        self.create_menu()

        self.rows = []

        # Set up the scrollable canvas
        self.scrollArea = QScrollArea(self)
        self.setCentralWidget(self.scrollArea)
        self.scrollArea.setWidgetResizable(True)

        # Container widget for the scroll area
        self.container = QWidget()
        self.scrollArea.setWidget(self.container)

        # Primary QVBoxLayout for the container
        self.layout = QVBoxLayout(self.container)
        self.layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)

    def create_menu(self):
        menu_bar = self.menuBar()

        new_menu = menu_bar.addMenu("New Graph")
        new_action_empty = new_menu.addAction("Empty")
        new_action_empty.triggered.connect(lambda: self.new_graph_row(nx.empty_graph(0, create_using=nx.Graph)))
        new_submenu_complete = new_menu.addMenu("Complete")
        for i in range(2, 11):
            new_action_complete = new_submenu_complete.addAction(f"K_{i}")
            new_action_complete.triggered.connect(lambda checked, ii=i: self.new_graph_row(nx.complete_graph(ii)))
        new_submenu_wheel = new_menu.addMenu("Wheel")
        for i in range(3, 12):
            new_action_wheel = new_submenu_wheel.addAction(f"W_{i}")
            new_action_wheel.triggered.connect(lambda checked, ii=i: self.new_graph_row(nx.wheel_graph(ii+1)))
        new_submenu_bipartite = new_menu.addMenu("Bipartite")
        for i in range(2, 6):
            for j in range(2, i+1):
                new_action_bipartite = new_submenu_bipartite.addAction(f"K_{{{i}, {j}}}")
                new_action_bipartite.triggered.connect(lambda checked, ii=i, jj=j:
                                                       self.new_graph_row(nx.complete_multipartite_graph(ii, jj)))

    def add_row(self, row: Row):
        row.set_row_index(len(self.rows))
        self.layout.addLayout(row.layout)
        self.rows.append(row)

    def new_graph_row(self, graph=None, graph_w_pos=None):
        if graph_w_pos is not None:
            row = Row(self, None, None, GraphExpression(graph_w_pos))
        else:
            row = Row(self, None, None, GraphExpression(GraphWithPos(graph)))
        self.add_row(row)

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
