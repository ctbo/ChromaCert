# ChromaCert (c) 2023 by Harald Bögeholz

import sys
import random

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


class GraphWidget(QWidget):
    def __init__(self, parent=None, graph=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)  # Set a fixed size for each graph widget

        # Initialize graph and selected nodes set
        self.G = graph if graph is not None else nx.wheel_graph(8)
        self.pos = nx.spring_layout(self.G)
        self.selected_nodes = set()
        self.dragging = False  # Flag to check if we are dragging a vertex
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

    def draw_graph(self):
        self.ax.clear()
        node_colors = [
            'red' if node in self.selected_nodes else 'blue' for node in self.G.nodes()
        ]
        nx.draw(self.G, pos=self.pos, ax=self.ax, with_labels=False, node_color=node_colors, node_size=100)
        self.canvas.draw()

    def on_press(self, event):
        # Ensure that only a left-click initiates a drag
        if event.button != 1:
            return

        # Check if a node was clicked
        nodes = list(self.G.nodes())
        if nodes:
            data = [self.pos[node] for node in nodes]
            data_x, data_y = zip(*data)
            distances = np.sqrt((data_x - event.xdata)**2 + (data_y - event.ydata)**2)

            # If a node is close enough to the click, consider it selected
            if min(distances) < 0.1:  # adjust this threshold if necessary
                i = np.argmin(distances)
                self.dragging = True
                self.dragged_node = nodes[i]

    def on_motion(self, event):
        if self.dragging and self.dragged_node is not None:
            self.drag_occurred = True
            self.pos[self.dragged_node] = (event.xdata, event.ydata)
            self.draw_graph()

    def on_release(self, event):
        # If dragging didn't occur, toggle the node's state
        if not self.drag_occurred and self.dragged_node is not None:
            if self.dragged_node in self.selected_nodes:
                self.selected_nodes.remove(self.dragged_node)
            else:
                self.selected_nodes.add(self.dragged_node)
            self.draw_graph()

        self.dragging = False
        self.dragged_node = None
        self.drag_occurred = False  # Reset the drag_occurred flag

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)

        createNodeAction = contextMenu.addAction("Create Vertex")
        createNodeAction.triggered.connect(lambda: self.option_create_node(event.pos()))

        toggleEdgeAction = contextMenu.addAction("Toggle Edge")
        toggleEdgeAction.triggered.connect(self.option_toggle_edge)
        if len(self.selected_nodes) != 2:
            toggleEdgeAction.setEnabled(False)

        deleteNodesAction = contextMenu.addAction("Delete Vertices")
        deleteNodesAction.triggered.connect(self.option_delete_nodes)
        if not self.selected_nodes:
            deleteNodesAction.setEnabled(False)

        springLayoutAction = contextMenu.addAction("Spring Layout")
        springLayoutAction.triggered.connect(self.option_spring_layout)

        # Show the context menu at the cursor's position
        contextMenu.exec(event.globalPos())

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
        new_node = max(self.G.nodes())+1 if self.G.nodes() else 0
        self.G.add_node(new_node)
        self.pos[new_node] = data_pos

        # Redraw the graph
        self.draw_graph()

    def option_toggle_edge(self):
        if len(self.selected_nodes) == 2:
            u, v = self.selected_nodes
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
            else:
                self.G.add_edge(u, v)
            self.draw_graph()  # Redraw the graph to reflect the changes

    def option_delete_nodes(self):
        for node in self.selected_nodes:
            self.G.remove_node(node)
            del self.pos[node]
        self.selected_nodes = set()
        self.draw_graph()

    def option_spring_layout(self):
        self.pos = nx.spring_layout(self.G)
        self.draw_graph()


class RowLabel(QLabel):
    def __init__(self, text, main_window=None, row_index=None, *args, **kwargs):
        super().__init__(text, *args, **kwargs)
        self.main_window = main_window
        self.row_index = row_index
        font = self.font()
        font.setBold(True)
        self.setFont(font)

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)

        # Sample action
        sampleAction = contextMenu.addAction("Add Row")
        sampleAction.triggered.connect(self.on_sample_action)

        # Display the context menu
        contextMenu.exec(event.globalPos())

    def on_sample_action(self):
        print(f"Sample action triggered! {self.row_index=}")
        if self.main_window:
            self.main_window.add_row()


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
        menuBar = self.menuBar()

        new_menu = menuBar.addMenu("New Graph")
        new_action_empty = new_menu.addAction("Empty")
        new_action_empty.triggered.connect(lambda: self.add_row(nx.empty_graph(0, create_using=nx.Graph)))
        new_submenu_complete = new_menu.addMenu("Complete")
        for i in range(2, 11):
            new_action_complete = new_submenu_complete.addAction(f"K_{i}")
            new_action_complete.triggered.connect(lambda checked, i=i: self.add_row(nx.complete_graph(i)))
        new_submenu_wheel = new_menu.addMenu("Wheel")
        for i in range(3, 11):
            new_action_wheel = new_submenu_wheel.addAction(f"W_{i}")
            new_action_wheel.triggered.connect(lambda checked, i=i: self.add_row(nx.wheel_graph(i+1)))
        new_submenu_bipartite = new_menu.addMenu("Bipartite")
        for i in range(2, 6):
            for j in range(2, i+1):
                new_action_bipartite = new_submenu_bipartite.addAction(f"K_{{{i}, {j}}}")
                new_action_bipartite.triggered.connect(lambda checked, i=i, j=j:
                                                       self.add_row(nx.complete_multipartite_graph(i, j)))

    def add_row(self, graph=None):
        row_number = len(self.rows) + 1
        hbox = QHBoxLayout()
        row_label = RowLabel(f"({row_number})", main_window=self, row_index=row_number-1)
        hbox.addWidget(row_label)

        if graph is not None:
            graph_widget = GraphWidget(self.container, graph=graph)
            hbox.addWidget(graph_widget)
        else:
            num_graphs = random.randint(1, 3)  # Random number of graphs between 1 and 3 for demonstration
            for _ in range(num_graphs):
                graph_widget = GraphWidget(self.container)
                hbox.addWidget(graph_widget)
                hbox.addWidget(QLabel(","))  # just to show that we can put text between graphs

        hbox.addStretch(1)  # push widgets to the left in each row

        self.layout.addLayout(hbox)
        self.rows.append(hbox)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
