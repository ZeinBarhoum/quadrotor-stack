
from qt_gui.plugin import Plugin
from .plan_command import QuadPlanCommandWidget1


class QuadPlanCommandPlugin1(Plugin):
    def __init__(self, context):
        super().__init__(context)

        self._widget = QuadPlanCommandWidget1(node=context.node)
        context.add_widget(self._widget)
