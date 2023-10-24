# ChromaCert (c) 2023 by Harald Bögeholz

import sys
import random

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QMenu

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)


class GraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)  # Set a fixed size for each graph widget

        # Initialize graph and selected nodes set
        self.G = nx.wheel_graph(8)
        self.pos = nx.spring_layout(self.G)
        self.selected_nodes = set()

        # Create a matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # Draw the graph initially
        self.draw_graph()

        # Connect the canvas click event to our custom function
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def draw_graph(self):
        self.ax.clear()
        node_colors = [
            'red' if node in self.selected_nodes else 'blue' for node in self.G.nodes()
        ]
        nx.draw(self.G, pos=self.pos, ax=self.ax, with_labels=True, node_color=node_colors)
        self.canvas.draw()

    def on_click(self, event):
        # Check if a node was clicked
        data = [self.pos[node] for node in self.G.nodes()]
        data_x, data_y = zip(*data)
        distances = np.sqrt((data_x-event.xdata) ** 2+(data_y-event.ydata) ** 2)

        # If a node is close enough to the click, consider it selected
        if min(distances) < 0.1:  # adjust this threshold if necessary
            node = np.argmin(distances)
            if node in self.selected_nodes:
                self.selected_nodes.remove(node)
            else:
                self.selected_nodes.add(node)
            self.draw_graph()

    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)

        # Add some dummy actions
        action1 = contextMenu.addAction("Spring Layout")
        action1.triggered.connect(self.option_spring_layout)

        action2 = contextMenu.addAction("Option 2")
        action2.triggered.connect(self.dummy_option)

        action3 = contextMenu.addAction("Option 3")
        action3.triggered.connect(self.dummy_option)

        # Show the context menu at the cursor's position
        contextMenu.exec(event.globalPos())

    def dummy_option(self):
        print("Selected context menu option")

    def option_spring_layout(self):
        self.pos = nx.spring_layout(self.G)
        self.draw_graph()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('ChromaCert')
        self.setGeometry(100, 100, 800, 600)

        self.create_menu()

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

        # Add a few rows as a starting point
        for _ in range(3):  # Add 3 rows for demonstration
            self.add_row()

        # Add a button to add more rows (for testing)
        self.addButton = QPushButton("Add Row", self.container)
        self.addButton.clicked.connect(self.add_row)
        self.layout.addWidget(self.addButton)  # Place button below the last row

    def create_menu(self):
        menuBar = self.menuBar()
        actionMenu = menuBar.addMenu('Actions')

        # Add some dummy functions
        action1 = actionMenu.addAction('Add row')
        action1.triggered.connect(self.add_row)

        action2 = actionMenu.addAction('Function 2')
        action2.triggered.connect(self.dummy_function)

        action3 = actionMenu.addAction('Function 3')
        action3.triggered.connect(self.dummy_function)

    def dummy_function(self):
        print("Dummy function executed")

    def add_row(self):
        hbox = QHBoxLayout()
        num_graphs = random.randint(1, 3)  # Random number of graphs between 1 and 3 for demonstration
        for _ in range(num_graphs):
            graph_widget = GraphWidget(self.container)
            hbox.addWidget(graph_widget)
            hbox.addWidget(QLabel(",")) # just to show that we can put text between graphs

        hbox.addStretch(1) # push widgets to the left in each row

        self.layout.addLayout(hbox)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
