import remi.gui as gui
from remi import start, App

class MyApp(App):
    def main(self):
        # Create a simple label
        label = gui.Label("Hello, Remi!", style={'font-size': '30px', 'margin': '20px'})

        # The root widget is returned
        return label

# start() runs the web server and opens browser
start(MyApp, address='0.0.0.0', port=8081, multiple_instance=False, enable_file_cache=True)
