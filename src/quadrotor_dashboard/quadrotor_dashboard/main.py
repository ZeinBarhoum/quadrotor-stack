import sys

from rqt_gui.main import Main


def main():
    main = Main()
    sys.exit(main.main(sys.argv, standalone='quadrotor_dashboard.plan_command_plugin.QuadPlanCommandPlugin'))


if __name__ == '__main__':
    main()
