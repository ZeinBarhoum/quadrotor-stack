from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QWidget, QTextEdit, QPushButton, QApplication

import rclpy
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import Point

import os


class QuadPlanCommandWidget1(QWidget):
    textEdit_x: QTextEdit
    textEdit_y: QTextEdit
    textEdit_z: QTextEdit
    pushButton_command: QPushButton

    def __init__(self, node):
        super().__init__()

        self._node = node

        self._publisher = self._node.create_publisher(msg_type=Point,
                                                      topic='quadrotor_plan_command',
                                                      qos_profile=10)

        ui_file = os.path.join(get_package_share_directory('quadrotor_dashboard'),
                               'resource',
                               'plan_command.ui')

        loadUi(ui_file, self)

        self.pushButton_command.clicked.connect(self.on_push_button_clicked)

    def on_push_button_clicked(self):
        point = Point()
        point.x = float(self.textEdit_x.toPlainText())
        point.y = float(self.textEdit_y.toPlainText())
        point.z = float(self.textEdit_z.toPlainText())
        self._node.get_logger().info('Publishing plan command: {}'.format(point))
        self._publisher.publish(point)


# app = QApplication([])
# widget = QuadPlanCommandWidget()
# widget.show()
# app.exec()
