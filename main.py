# ChromaCert
# (c) 2023 by Harald Bögeholz

import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWidgets import QPushButton, QMenuBar, QMenu
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)


class GraphApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a wheel graph
        self.G = nx.wheel_graph(8)
        self.pos = nx.spring_layout(self.G)
        self.selected_nodes = set()

        # Setup the main window
        self.setWindowTitle('ChromaCert')
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        layout = QVBoxLayout(self.main_widget)

        # Create a matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)

        # Draw the graph initially
        self.draw_graph()

        # Connect the canvas click event to our custom function
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Add Menu
        self.add_menu()

        # Add Buttons
        self.add_buttons(layout)

    def add_menu(self):
        menuBar = self.menuBar()
        functionsMenu = menuBar.addMenu('Functions')

        action1 = functionsMenu.addAction('Function 1')
        action1.triggered.connect(self.function1)

        action2 = functionsMenu.addAction('Function 2')
        action2.triggered.connect(self.function2)

    def add_buttons(self, layout):
        button1 = QPushButton('Function 1', self)
        button1.clicked.connect(self.function1)
        layout.addWidget(button1)

        button2 = QPushButton('Function 2', self)
        button2.clicked.connect(self.function2)
        layout.addWidget(button2)

    def function1(self):
        print("Function 1 executed")
        self.pos = nx.spring_layout(self.G)
        self.draw_graph()

    def function2(self):
        print("Function 2 executed")

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = GraphApp()
    mainWin.show()
    sys.exit(app.exec_())
