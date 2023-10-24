import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtWidgets import QLabel

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
        self.layout = QVBoxLayout(self.container)
        self.layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinAndMaxSize)  # Add this line

        # Add a few graph widgets as a starting point
        for _ in range(5):
            graph_widget = GraphWidget(self.container)
            self.layout.addWidget(graph_widget)

        # Add a button to add more graph widgets (for testing)
        self.addButton = QPushButton("Add Graph", self.container)
        self.addButton.clicked.connect(self.add_graph_widget)
        self.layout.addWidget(self.addButton)

    def add_graph_widget(self):
        graph_widget = GraphWidget(self.container)
        self.layout.addWidget(graph_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
