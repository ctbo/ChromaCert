import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QHBoxLayout

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


    def add_row(self):
        hbox = QHBoxLayout()
        num_graphs = random.randint(1, 3)  # Random number of graphs between 1 and 3 for demonstration
        for _ in range(num_graphs):
            graph_widget = GraphWidget(self.container)
            hbox.addWidget(graph_widget)
            hbox.addWidget(QLabel("*"*random.randint(1,50)))

        hbox.addStretch(1) # push widgets to the left in each row

        self.layout.addLayout(hbox)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
