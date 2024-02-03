from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QPushButton, QWidget, QTableWidgetItem, QTableWidget, QStyledItemDelegate, QLineEdit, QFileDialog
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from qt_gui.plugin import Plugin
import sys
from rclpy.node import Node
from rqt_gui.main import Main
from ament_index_python.packages import get_package_share_directory
import os
from quadrotor_interfaces.msg import PathWayPoints
from geometry_msgs.msg import Point
import pandas as pd


class NumericDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = super(NumericDelegate, self).createEditor(parent, option, index)
        if isinstance(editor, QLineEdit):
            reg_ex = QRegExp("[+-]?[0-9]+.?[0-9]{,2}")
            validator = QRegExpValidator(reg_ex, editor)
            editor.setValidator(validator)
        return editor


class QuadPublishWaypointsWidget(QWidget):
    TB_waypoints: QTableWidget
    PB_new: QPushButton
    PB_remove: QPushButton
    PB_publish: QPushButton
    PB_load: QPushButton
    PB_save: QPushButton
    node: Node

    def __init__(self, node):
        super().__init__()
        ui_file = os.path.join(get_package_share_directory('quadrotor_dashboard'),
                               'resource',
                               'plugin_publish_waypoints.ui')

        loadUi(ui_file, self)

        delegate = NumericDelegate(self.TB_waypoints)
        self.TB_waypoints.setItemDelegate(delegate)

        self.node = node
        self.publisher = self.node.create_publisher(PathWayPoints, '/quadrotor_waypoints', 10)

        self.PB_new.clicked.connect(self.PB_new_clicked)
        self.PB_remove.clicked.connect(self.PB_remove_clicked)
        self.PB_publish.clicked.connect(self.PB_publish_clicked)
        self.PB_save.clicked.connect(self.PB_save_clicked)
        self.PB_load.clicked.connect(self.PB_load_clicked)

    def PB_new_clicked(self):
        self.TB_waypoints.insertRow(self.TB_waypoints.rowCount())
        for i in range(self.TB_waypoints.colorCount()):
            self.TB_waypoints.setItem(self.TB_waypoints.rowCount()-1, i, QTableWidgetItem('0'))

    def PB_remove_clicked(self):
        self.TB_waypoints.removeRow(self.TB_waypoints.currentRow())

    def PB_publish_clicked(self):
        print("Publishing...")
        msg = PathWayPoints()
        msg.waypoints = []
        msg.heading_angles = []
        for row in range(self.TB_waypoints.rowCount()):
            point = Point()
            point.x = float(self.TB_waypoints.item(row, 0).text())
            point.y = float(self.TB_waypoints.item(row, 1).text())
            point.z = float(self.TB_waypoints.item(row, 2).text())
            msg.waypoints.append(point)
            msg.heading_angles.append(float(self.TB_waypoints.item(row, 3).text()))
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        self.publisher.publish(msg)

    def PB_save_clicked(self):
        print("Saving...")
        file_names = QFileDialog.getSaveFileName(self, "Select File", "", "*.csv")
        try:
            file_name = file_names[0]
            if (not file_name.endswith('.csv')):
                file_name += '.csv'

            df = pd.DataFrame(columns=['x', 'y', 'z', 'psi'])
            for row in range(self.TB_waypoints.rowCount()):
                x = float(self.TB_waypoints.item(row, 0).text())
                y = float(self.TB_waypoints.item(row, 1).text())
                z = float(self.TB_waypoints.item(row, 2).text())
                psi = float(self.TB_waypoints.item(row, 3).text())
                df.loc[row] = [x, y, z, psi]
            self.node.get_logger().info(f'Saving waypoints to: {file_name}')
            df.to_csv(file_name, index=False)
        except (IsADirectoryError, FileNotFoundError) as e:
            self.node.get_logger().info('No file selected')

    def PB_load_clicked(self):
        print("Loading...")
        file_names = QFileDialog.getOpenFileName(self, "Select File", "", "*.csv")
        try:
            file_name = file_names[0]
            if (not file_name.endswith('.csv')):
                file_name += '.csv'
            df = pd.read_csv(file_name)
            self.TB_waypoints.setRowCount(0)
            for row in range(df.shape[0]):
                self.TB_waypoints.insertRow(self.TB_waypoints.rowCount())
                for col in range(df.shape[1]):
                    self.TB_waypoints.setItem(self.TB_waypoints.rowCount()-1, col, QTableWidgetItem(str(df.iloc[row, col])))
            self.node.get_logger().info(f'Loading waypoints from: {file_name}')
        except (IsADirectoryError, FileNotFoundError) as e:
            self.node.get_logger().info('No file selected Or BAD FILE')


class QuadPublishWaypointsPlugin(Plugin):
    def __init__(self, context):
        super().__init__(context)
        self._widget = QuadPublishWaypointsWidget(node=context.node)
        context.add_widget(self._widget)


def main():
    main = Main()
    sys.exit(main.main(sys.argv, standalone='quadrotor_dashboard.plugin_publish_waypoints.QuadPublishWaypointsPlugin'))


if __name__ == '__main__':
    main()
