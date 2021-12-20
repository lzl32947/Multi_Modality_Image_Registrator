import base64
import io
import os
import PySimpleGUI as sg
import cv2
import tempfile

import numpy as np
from PIL import Image, ImageDraw


class AnnotationWindow(object):
    def __init__(self):
        self.annotation_path = "annotation"
        self.flir_path = r"K:\FLIR_ADAS_1_3\train\thermal_8_bit"
        self.rgb_path = r"K:\FLIR_ADAS_1_3\train\RGB"
        self.output_path = r"K:\FLIR_ADAS_1_3\train\adjusted"
        self.file_types = [("JPEG (*.jpg)", "*.jpg"), ("All files (*.*)", "*.*")]
        self.tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg").name
        self.layout = self.load_layout()
        self.check_annotations()
        self.window = None
        self.dragging = False
        self.start_point = self.end_point = self.prior_rect = None
        self.now_image = None
        self.abs_path = None
        self.file_list = None
        self.drag_figures = None
        self.last_x_and_y = None
        self.image_id = None
        self.line_dict = None
        self.prev_line_dict = None
        self.H = None
        self.image_path = None
        self.current_rgb = None
        self.current_inf = None
        self.original_rgb = None
        self.output_image = None
        self.x_float = 1
        self.rgb_ratio = 0.5
        self.inf_ratio = 0.5
        self.y_float = 1
        self.lock_input = False

    def check_annotations(self):
        if not os.path.exists(self.annotation_path):
            os.mkdir(self.annotation_path)

    def load_layout(self):
        program_layout = [
            [sg.Text(key='-INFO-', size=(60, 1))],
            [
                sg.Graph(
                    canvas_size=(1280, 512),
                    graph_bottom_left=(0, 512),
                    graph_top_right=(1280, 0),
                    key="-GRAPH-",
                    enable_events=True,
                    drag_submits=True,
                    right_click_menu=[[], ['Erase item', 'Erase all']]
                ), sg.Image(key="-IMAGE-")
            ],
            [
                sg.Text("Image File"),
                sg.Input(
                    size=(40, 1), key="-FILENAME-"
                ),
                sg.FileBrowse(file_types=self.file_types, key="-BROWSER-"),
                sg.Button("Load Image"),
            ],

            [sg.Button("Previous Image"), sg.Button("Next Image")],
            [sg.Text("RGB ratio"), sg.Input(default_text="0.5",
                                            size=(5, 1), key="-RGB-"
                                            ), sg.Text("INF ratio"), sg.Input(
                size=(5, 1), key="-INF-", default_text="0.5"
            ), sg.Button("Set Ratio"), sg.Button("(Un)Lock Ratio")],
            [sg.Button("Calculate H"), sg.Button("Load Previous H"), sg.Button("Use Previous H")],
            [sg.Button("Export")],
        ]
        return program_layout

    def load_image_only(self, path):
        image_name = path.split("/")[-1].split(".")[0]
        other_image = os.path.join(self.flir_path, image_name + ".jpeg")
        if os.path.exists(path) and os.path.exists(other_image):
            image = Image.open(path)
            self.original_rgb = np.array(image, dtype=np.uint8).copy()
            self.x_float = image.size[0] / 640
            self.y_float = image.size[1] / 512
            image = image.resize((640, 512))
            flir_image = Image.open(other_image)
            image_np = np.array(image, dtype=np.uint8)
            self.current_rgb = image_np.copy()
            flir_image_np = np.array(flir_image, dtype=np.uint8)
            flir_image_np = np.expand_dims(flir_image, axis=-1)
            flir_image_np = np.concatenate([flir_image_np, flir_image_np, flir_image_np], axis=-1)
            self.current_inf = flir_image_np.copy()
            concat = np.concatenate([image_np, flir_image_np], axis=1)
            im_pil = Image.fromarray(concat)
            with io.BytesIO() as output:
                im_pil.save(output, format="PNG")
                data = output.getvalue()
            im_64 = base64.b64encode(data)
            self.image_id = self.window['-GRAPH-'].draw_image(data=im_64, location=(0, 0))

    def load_annotation_only(self, path):
        graph = self.window["-GRAPH-"]
        path = path + ".txt"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin.readlines():
                    rx, ry, ix, iy = line.replace("\n", "").split(",")
                    rx, ry, ix, iy = int(rx), int(ry), int(ix), int(iy)
                    self.prior_rect = graph.draw_line((rx, ry), (ix, iy), width=4)
                    if self.line_dict is None:
                        self.line_dict = dict()
                    self.line_dict[self.prior_rect] = [(rx, ry), (ix, iy)]
                self.window["-INFO-"].update(value="load annotations")
        if self.line_dict is not None:
            self.calculate_H()
        self.start_point, self.end_point = None, None  # enable grabbing a new rect
        self.dragging = False
        self.prior_rect = None

    def save_annotation(self):
        if self.line_dict is not None:
            save_path = os.path.join(self.annotation_path, self.now_image + ".txt")
            with open(save_path, "w", encoding="utf-8") as fout:
                if len(self.line_dict.keys()) == 4:
                    for item in self.line_dict.keys():
                        fout.write("{},{},{},{}\n".format(self.line_dict[item][0][0], self.line_dict[item][0][1],
                                                          self.line_dict[item][1][0], self.line_dict[item][1][1]))
            self.prev_line_dict = self.line_dict.copy()
            self.line_dict = None

    def load_image_and_annotation(self, image_path):
        self.window["-FILENAME-"].update(value=image_path, disabled=True)
        self.now_image = image_path.split("/")[-1]
        self.abs_path = image_path.replace(self.now_image, "")
        self.file_list = os.listdir(self.abs_path)
        self.load_image_only(os.path.join(self.abs_path, self.now_image))
        self.load_annotation_only(os.path.join(self.annotation_path, self.now_image))

    def load_previous_H(self):
        if self.prev_line_dict is not None and len(self.prev_line_dict) == 4:
            rgb_points = []
            flir_points = []
            for item in self.prev_line_dict.keys():
                fx, fy, bx, by = self.prev_line_dict[item][0][0], self.prev_line_dict[item][0][1], \
                                 self.prev_line_dict[item][1][0], \
                                 self.prev_line_dict[item][1][1]
                if fx < 640:
                    rx, ry, ix, iy = fx, fy, bx, by
                else:
                    rx, ry, ix, iy = bx, by, fx, fy
                ix -= 640
                rx = int(rx * self.x_float)
                ry = int(ry * self.y_float)
                rgb_points.append([rx, ry])
                flir_points.append([ix, iy])
            rgb_features = np.array(rgb_points, dtype=np.float32)
            infra_red_features = np.array(flir_points, dtype=np.float32)
            try:
                self.H = cv2.getPerspectiveTransform(rgb_features, infra_red_features)
                self.output_image = cv2.warpPerspective(self.original_rgb, self.H,
                                                        (self.current_inf.shape[1], self.current_inf.shape[0]))
                self.load_to_photo(self.output_image)
            except cv2.error as e:
                self.window["-INFO-"].update(value=e.msg)

    def use_previous_H(self):
        self.line_dict = self.prev_line_dict.copy()

    def draw_line(self, value):
        graph = self.window["-GRAPH-"]
        x, y = value["-GRAPH-"]
        if not self.dragging:
            self.start_point = (x, y)
            self.dragging = True
            self.drag_figures = graph.get_figures_at_location((x, y))
            self.last_x_and_y = x, y
        else:
            self.end_point = (x, y)
        if self.prior_rect:
            graph.delete_figure(self.prior_rect)
        self.last_x_and_y = x, y
        if None not in (self.start_point, self.end_point):
            self.prior_rect = graph.draw_line(self.start_point, self.end_point, width=4)
            self.window["-INFO-"].update(value=f"mouse {value['-GRAPH-']}")

    def draw_line_finish(self, value):
        self.window["-INFO-"].update(value=f"grabbed rectangle from {self.start_point} to {self.end_point}")
        if self.line_dict is None:
            self.line_dict = dict()
        self.line_dict[self.prior_rect] = [self.start_point, self.end_point]
        self.start_point, self.end_point = None, None  # enable grabbing a new rect
        self.dragging = False
        self.prior_rect = None

    def erase_element(self, value):
        graph = self.window["-GRAPH-"]
        self.window["-INFO-"].update(value=f"Right click erase at {value['-GRAPH-']}")
        if value['-GRAPH-'] != (None, None):
            drag_figures = graph.get_figures_at_location(value['-GRAPH-'])
            for figure in drag_figures:
                if figure == self.image_id:
                    continue
                else:
                    graph.delete_figure(figure)
                    if self.line_dict is not None:
                        if figure in self.line_dict.keys():
                            del self.line_dict[figure]

    def next_image(self):
        if self.now_image is not None:
            image_id = self.file_list.index(self.now_image)
            if image_id < len(self.file_list) - 1:
                self.save_annotation()
                self.image_path = os.path.join(self.abs_path, self.file_list[image_id + 1])
                self.load_image_and_annotation(self.image_path)
                self.now_image = self.file_list[image_id + 1]
                self.output_image = None

    def prev_image(self):
        if self.now_image is not None:
            image_id = self.file_list.index(self.now_image)
            if image_id > 0:
                self.save_annotation()
                self.image_path = os.path.join(self.abs_path, self.file_list[image_id - 1])
                self.load_image_and_annotation(self.image_path)
                self.now_image = self.file_list[image_id - 1]
                self.output_image = None

    def run(self):
        self.window = sg.Window("Drawing GUI", self.layout, size=(1920, 740), return_keyboard_events=True)
        while True:
            event, values = self.window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            elif event == "Load Image":
                if values["-FILENAME-"] != "":
                    self.image_path = values["-FILENAME-"]
                    self.load_image_and_annotation(self.image_path)
            elif event == "-GRAPH-":
                self.draw_line(values)
            elif event.endswith('+UP'):  # The drawing has ended because mouse up
                self.draw_line_finish(values)
            elif event.endswith('+RIGHT+'):  # Righ click
                self.window["-INFO-"].update(value=f"Right clicked location {values['-GRAPH-']}")
            elif event.endswith('+MOTION+'):  # Righ click
                self.window["-INFO-"].update(value=f"mouse freely moving {values['-GRAPH-']}")
            elif event == 'Erase item':
                self.erase_element(values)
            elif event == 'Erase all':
                self.erase_all(values)
            elif event == "d" or event == "Next Image":
                self.next_image()
            elif event == "a" or event == "Previous Image":
                self.prev_image()
            elif event == "Calculate H" or event == "s":
                self.calculate_H()
            elif event == "Load Previous H" or event == "w":
                self.load_previous_H()
            elif event == "Use Previous H" or event == "e":
                self.use_previous_H()
            elif event == "Export" or event == "f":
                self.export_image()
            elif event == "Set Ratio":
                self.set_ratio(values)
                self.calculate_H()
            elif event == "(Un)Lock Ratio":
                self.window['-INF-'].update(disabled=self.lock_input)
                self.window['-RGB-'].update(disabled=self.lock_input)
                self.lock_input = not self.lock_input
        self.window.close()

    def load_to_photo(self, image):
        rolling_image = self.current_inf * self.inf_ratio + image * self.rgb_ratio
        im_pil = rolling_image.astype(np.uint8)
        im_pil = Image.fromarray(im_pil)
        with io.BytesIO() as output:
            im_pil.save(output, format="PNG")

            self.window["-IMAGE-"].update(data=output.getvalue())

    def calculate_H(self):

        if self.line_dict is not None and len(self.line_dict.keys()) == 4:
            rgb_points = []
            flir_points = []
            for item in self.line_dict.keys():
                fx, fy, bx, by = self.line_dict[item][0][0], self.line_dict[item][0][1], self.line_dict[item][1][0], \
                                 self.line_dict[item][1][1]
                if fx < 640:
                    rx, ry, ix, iy = fx, fy, bx, by
                else:
                    rx, ry, ix, iy = bx, by, fx, fy
                ix -= 640
                rx = int(rx * self.x_float)
                ry = int(ry * self.y_float)
                rgb_points.append([rx, ry])
                flir_points.append([ix, iy])
            rgb_features = np.array(rgb_points, dtype=np.float32)
            infra_red_features = np.array(flir_points, dtype=np.float32)
            try:
                self.H = cv2.getPerspectiveTransform(rgb_features, infra_red_features)
                self.output_image = cv2.warpPerspective(self.original_rgb, self.H,
                                                        (self.current_inf.shape[1], self.current_inf.shape[0]))
                self.load_to_photo(self.output_image)
            except cv2.error as e:
                self.window["-INFO-"].update(value=e.msg)

    def erase_all(self, value):
        graph = self.window["-GRAPH-"]
        graph.erase()
        self.line_dict = None
        self.load_image_only(self.image_path)
        self.dragging = False

    def export_image(self):
        if self.output_image is not None:
            image_name = self.image_path.split("/")[-1].split(".")[0]
            cv2.imwrite(os.path.join(self.output_path, image_name + ".jpg"), self.output_image)
            self.window["-INFO-"].update(value=r"output {}.jpg complete!".format(image_name))

    def set_ratio(self, values):
        rgb_ratio = values["-RGB-"]
        inf_ratio = values["-INF-"]
        try:
            rgb_ratio = min(max(float(rgb_ratio), 0), 1)
            inf_ratio = min(max(float(inf_ratio), 0), 1)
            if rgb_ratio + inf_ratio != 1:
                raise ValueError
            self.rgb_ratio = rgb_ratio
            self.inf_ratio = inf_ratio
        except ValueError:
            self.window["-INFO-"].update(value="Value error in set ratio!")


if __name__ == '__main__':
    AnnotationWindow().run()
