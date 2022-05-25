from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from obj_img import run_img
from obj_vid import run_vid


class TrafficPrediction(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label = None
        self.img_button = None
        self.file_path_input = None
        self.window = None
        self.vid_button = None

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

        img_button = Button(text='run img')
        img_button.bind(on_press=self.write_wunning,
                        on_release=self.img_button_click)
        window.add_widget(img_button)
        self.img_button = img_button

        vid_button = Button(text='run vid')
        vid_button.bind(on_press=self.write_wunning,
                        on_release=self.vid_button_click)
        window.add_widget(vid_button)
        self.vid_button = vid_button

        label = Label()
        window.add_widget(label)
        self.label = label

        return window

    def write_wunning(self, instance):
        if len(self.file_path_input.text) > 0:
            self.label.text = 'running'
        else:
            self.label.text = 'bad path'

    def img_button_click(self, instance):
        if len(self.file_path_input.text) > 0:
            run_img(self.file_path_input.text)
            self.label.text = 'done'
        else:
            self.label.text = 'bad path'

    def vid_button_click(self, instance):
        if len(self.file_path_input.text) > 0:
            run_vid(self.file_path_input.text)
            self.label.text = 'done'
        else:
            self.label.text = 'bad path'


if __name__ == "__main__":
    TrafficPrediction().run()
