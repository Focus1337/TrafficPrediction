from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from obj_img import run
import asyncio
from kivy.app import async_runTouchApp


class TrafficPrediction(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label = None
        self.button = None
        self.file_path_input = None
        self.window = None

    def build(self):
        window = GridLayout()
        window.cols = 1
        window.size_hint = (0.8, 0.8)
        window.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        self.window = window

        file_path_input = TextInput(multiline=False,
                                    size_hint=(1, 0.2))
        window.add_widget(file_path_input)
        self.file_path_input = file_path_input

        button = Button(text='run')
        button.bind(on_press=self.write_wunning,
                    on_release=self.button_click)
        window.add_widget(button)
        self.button = button

        label = Label()
        window.add_widget(label)
        self.label = label

        return window

    def write_wunning(self, instance):
        if len(self.file_path_input.text) > 0:
            self.label.text = 'running'
        else:
            self.label.text = 'bad path'

    def button_click(self, instance):
        if len(self.file_path_input.text) > 0:
            run(self.file_path_input.text)
            self.label.text = 'done'
        else:
            self.label.text = 'bad path'


if __name__ == "__main__":
    TrafficPrediction().run()
