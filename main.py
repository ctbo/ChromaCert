import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QGridLayout

import random

class GraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)  # Set a fixed size for each graph widget for now

        # Add a layout and label to make the widget more visible
        layout = QVBoxLayout(self)
        label = QLabel("Graph Placeholder", self)
        layout.addWidget(label)

        # Set border and background
        self.setStyleSheet("border: 2px solid black; background-color: lightgray;")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('ChromaCert')
        self.setGeometry(100, 100, 800, 600)

        # Set up the scrollable canvas
        self.scrollArea = QScrollArea(self)
        self.setCentralWidget(self.scrollArea)
        self.scrollArea.setWidgetResizable(True)

        # Container widget for the scroll area
        self.container = QWidget()
        self.scrollArea.setWidget(self.container)

        # Layout for the container
        self.layout = QGridLayout(self.container)
        self.layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)

        # Add a few rows of graph widgets as a starting point
        for _ in range(3):  # Add 3 rows for demonstration
            self.add_graph_row()

        # Add a button to add more rows of graph widgets (for testing)
        self.addButton = QPushButton("Add Graph Row", self.container)
        self.addButton.clicked.connect(self.add_graph_row)
        self.layout.addWidget(self.addButton, self.layout.rowCount(), 0)  # Place button below the last row


    def add_graph_row(self):
        row = self.layout.rowCount()
        num_graphs = random.randint(1, 5)  # Random number of graphs between 1 and 5
        for col in range(num_graphs):
            graph_widget = GraphWidget(self.container)
            self.layout.addWidget(graph_widget, row, col)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
