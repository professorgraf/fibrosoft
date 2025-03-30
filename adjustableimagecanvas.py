import math
import warnings
import tkinter as tk

import platform

from tkinter import ttk
from PIL import Image, ImageTk


class GridAutoScrollbar(ttk.Scrollbar):
    """ a grid based auto-disappearing scrollbar """
    def pack(self, **kw):
        raise tk.TclError('Grid mode only / pack not available for ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Grid mode only / place not available for ' + self.__class__.__name__)

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)


class Marker:
    def __init__(self, item, x, y):
        self.position = [x, y]
        self.item = item


class AdjustableImageCanvas(tk.Canvas):
    class MarkerSetting:
        def __init__(self, style='rectangle', color='blue'):
            self.style = style
            self.color = color

    """ Displays, drag-moves, and zooms in or out to a connected image,
        left mouse button is available for tools like setting markers """
    def __init__(self, parent, width=600, height=600, path=None):
        """ Initialize the ImageFrame """
        super().__init__(height=height, width=width)
        self.scalefactor = 1.0          # scale factor for the canvas image zoom
        self.__delta = 1.3              # zoom magnitude
        self.__filter = Image.LANCZOS   # NEAREST, BILINEAR, BICUBIC, or LANCZOS
        self.__wheel_delta = 119 if platform.system() == 'Windows' else 0  # Windows or Darwin (Linux has diff. method)
        self.__previous_state = 0       # previous state of the keyboard
        self.path = path                # in case image is already known while setup, usually: set_image is used

        self.command = None
        self.marker_radius_px = 10
        self.marker_settings = dict()
        self.marker_settings['default'] = self.MarkerSetting()
        self.marker_lists = dict()
        self.selected_marker = None
        self.modified = False

        # Create a permanent ImageFrame in parent widget
        self.__permanentframe = ttk.Frame(parent, width=width, height=height)
        # Vertical and horizontal scrollbars for canvas
        horz_scrollbar = GridAutoScrollbar(self.__permanentframe, orient='horizontal')
        vert_scrollbar = GridAutoScrollbar(self.__permanentframe, orient='vertical')
        horz_scrollbar.grid(row=1, column=0, sticky='we')
        vert_scrollbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__permanentframe, highlightthickness=0,
                                xscrollcommand=horz_scrollbar.set, yscrollcommand=vert_scrollbar.set, width=width, height=height)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        horz_scrollbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vert_scrollbar.configure(command=self.__scroll_y)

        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # check if the canvas was resized
        self.canvas.bind('<ButtonPress-2>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B2-Motion>',     self.__move_to)  # move canvas to the new position

        self.canvas.bind('<ButtonPress-1>', self.__on_mouse_press)
        self.canvas.bind('<B1-Motion>', self.__on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.__on_mouse_release)
        self.canvas.bind('<KeyRelease>', self.__keystroke)

        # add mouse wheel zoom functionality
        if platform.system() == "Linux":
            self.canvas.bind('<Button-5>', self.__on_mousewheel)  # scroll down
            self.canvas.bind('<Button-4>', self.__on_mousewheel)  # scroll up
        else:
            self.canvas.bind('<MouseWheel>', self.__on_mousewheel)  # mouse-wheel interactions for Windows and macOS

        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__is_huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for big images

        self.container = None
        self.__image = None
        self.__min_side = None
        self.__reduction = None
        self.__scale = None
        self.__curr_img = None
        self.__ratio = None
        self.imageheight = None
        self.imagewidth = None
        self.__offset = None
        self.__tile = None
        self.__pyramid = None
        if path is not None:
            self.set_image_file(self.path)

        # demo to show the circles and stuff is also resized
        # self.canvas.create_oval(20, 20, 100, 100, width=2, outline='red')

    def set_marker_style(self, name, style='rectangle', color='red'):
        self.marker_settings[name] = self.MarkerSetting(style, color)

    def set_image_file(self, file):
        self.path = file
        if self.__image is not None:
            self.canvas.delete(self.canvas.image_id)
            self.free_mem()
            self.scalefactor = 1
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)
        self.imagewidth, self.imageheight = self.__image.size
        if self.imagewidth * self.imageheight > self.__huge_size * self.__huge_size and \
           self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__is_huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it needs to be 'raw'
                           [0, 0, self.imagewidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imagewidth, self.imageheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__is_huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imagewidth, self.imageheight) / self.__huge_size if self.__is_huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.scalefactor * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imagewidth, self.imageheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas
        self.clear_all_markers()
        self.modified = False

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imagewidth), float(self.imageheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imageheight / self.__band_width)
        while i < self.imageheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imageheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imagewidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imagewidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imagewidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k)+1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30*' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__permanentframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__permanentframe.grid(sticky='nswe')  # make frame container sticky
        self.__permanentframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__permanentframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        # in this method we deal with the resizing matters
        if self.path is None:
            return  # quit if there is no image
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__is_huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.scalefactor)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imagewidth * int(y1 / self.scalefactor) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imagewidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.scalefactor), 0, int(x2 / self.scalefactor), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                                    (int(x1 / self.__scale), int(y1 / self.__scale),
                                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            if hasattr(self.canvas, 'image_id') and self.canvas.image_id is not None:
                self.canvas.delete(self.canvas.image_id)
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            #image_id = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
            #                                   max(box_canvas[1], box_img_int[1]),
            #                                   anchor='nw', image=imagetk)

            # create the image on the canvas, but never get outside the point of origin
            image_id = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                                max(box_canvas[1], box_img_int[1]),
                                                anchor='nw', image=imagetk)

            self.canvas.lower(image_id)  # set image into background
            self.canvas.image_id = image_id
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def is_inside(self, x, y):
        # is point (x,y) inside the image area
        bbox = self.canvas.coords(self.container)  # get image area
        return bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]

    def get_marker_on_canvas(self, x, y):
        for lists in self.marker_lists.values():
            for marker in lists:
                bbox = self.canvas.coords(marker.item)  # get image area
                if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
                    return marker # marker found
        return None # no marker found

    def clear_all_markers(self):
        for lists in self.marker_lists.values():
            for marker in lists:
                self.canvas.delete(marker.item)
        self.marker_lists.clear()
        self.modified = False

    def __on_mousewheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.path is None or not self.is_inside(x, y):
            return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta < -self.__wheel_delta:  # == -120:  # scroll down, smaller
            if round(self.__min_side * self.scalefactor) < round(min(self.canvas.winfo_height(), self.canvas.winfo_width())):
                return  # image is less than 30 pixels
            self.scalefactor /= self.__delta
            scale /= self.__delta
        if event.num == 4 or event.delta > self.__wheel_delta:  # == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.scalefactor:
                return  # 1 pixel is bigger than the visible area
            self.scalefactor *= self.__delta
            scale *= self.__delta
        # Take appropriate image from the pyramid
        k = self.scalefactor * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def zoom_out(self):
        if self.path is not None:
            x = self.canvas.canvasx( int(self.canvas.winfo_width()-self.canvas.imagetk.width()) >> 1 )
            y = self.canvas.canvasy( int(self.canvas.winfo_height()-self.canvas.imagetk.height()) >> 1 )
            self.scalefactor /= self.__delta
            scale = 1/self.__delta
            self.__zoom(scale, x, y)

    def zoom_in(self):
        if self.path is not None:
            x = self.canvas.canvasx( int(self.canvas.winfo_width()-self.canvas.imagetk.width()) >> 1 )
            y = self.canvas.canvasy( int(self.canvas.winfo_height()-self.canvas.imagetk.height()) >> 1 )
            self.scalefactor *= self.__delta
            scale = 1 * self.__delta
            self.__zoom(scale, x, y)

    def __zoom(self, scale, center_x, center_y):
        # Take appropriate image from the pyramid
        k = self.scalefactor * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', center_x, center_y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def set_command(self, command):
        self.command = command

    def __on_mouse_press(self, event):
        if self.__image is None or self.command is None:
            return

        container_rel_coords = self.canvas.coords(self.container)
        canvas_click = [self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)]
        pixel_pos = [canvas_click[0]-container_rel_coords[0], canvas_click[1]-container_rel_coords[1]]
        image_pos = [int(round(pixel_pos[0] / self.scalefactor, 0)), int(round(pixel_pos[1] / self.scalefactor, 0))]

        marker_radius = (self.marker_radius_px * self.scalefactor)
        if marker_radius < 2:
            marker_radius = 2

        if self.selected_marker is not None: # clear the inner filling when new press on canvas
            self.canvas.itemconfig(self.selected_marker.item, fill="")

        self.selected_marker = self.get_marker_on_canvas(canvas_click[0], canvas_click[1])

        if self.selected_marker is None:
            if self.command not in self.marker_lists:
                self.marker_lists[self.command] = []
            try:
                setting = None if self.command == "select" else self.marker_settings[self.command]
            except Exception:
                setting = None
            if setting is None:
                setting = self.marker_settings['default']

            if setting.style == 'rectangle':
                mark = self.canvas.create_rectangle(canvas_click[0] - marker_radius, canvas_click[1] - marker_radius,
                                                    canvas_click[0] + marker_radius, canvas_click[1] + marker_radius,
                                                    width=2, outline=setting.color)
            if setting.style == 'circle':
                mark = self.canvas.create_oval(canvas_click[0] - marker_radius, canvas_click[1] - marker_radius,
                                               canvas_click[0] + marker_radius, canvas_click[1] + marker_radius,
                                               width=2, outline=setting.color)
            marker = Marker(mark, *image_pos)
            self.marker_lists[self.command].append(marker)
            self.selected_marker = marker
            self.modified = True
        self.canvas.itemconfig(self.selected_marker.item, fill="gray50")

    def __on_mouse_drag(self, event):
        if self.__image is None or self.command is None:
            return
        marker_radius = (self.marker_radius_px * self.scalefactor)
        if marker_radius < 2:
            marker_radius = 2
        canvas_click = [self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)]
        container_rel_coords = self.canvas.coords(self.container)
        pixel_pos = [canvas_click[0]-container_rel_coords[0], canvas_click[1]-container_rel_coords[1]]
        image_pos = [int(round(pixel_pos[0] / self.scalefactor, 0)), int(round(pixel_pos[1] / self.scalefactor, 0))]

        if self.selected_marker is not None:        # (self.command == 'select' or self.command == 'drag') and
            self.canvas.moveto(self.selected_marker.item, canvas_click[0]-marker_radius, canvas_click[1]-marker_radius)
            self.selected_marker.position = image_pos
            self.modified = True

    def __on_mouse_release(self, event):
        if self.__image is None or self.command is None:
            return
        marker_radius = (self.marker_radius_px * self.scalefactor)
        if marker_radius < 2:
            marker_radius = 2
        canvas_click = [self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)]
        container_rel_coords = self.canvas.coords(self.container)
        pixel_pos = [canvas_click[0]-container_rel_coords[0], canvas_click[1]-container_rel_coords[1]]
        image_pos = [int(round(pixel_pos[0] / self.scalefactor, 0)), int(round(pixel_pos[1] / self.scalefactor, 0))]

        if self.selected_marker is not None:
            self.canvas.moveto(self.selected_marker.item, canvas_click[0]-marker_radius, canvas_click[1]-marker_radius)
            self.selected_marker.position = image_pos
            self.modified = True

    def delete_selected(self):
        if self.selected_marker is not None:
            for key in self.marker_lists.keys():
                if self.selected_marker in self.marker_lists[key]:
                    self.marker_lists[key] = [item for item in self.marker_lists[key] if item != self.selected_marker]
                    self.canvas.delete( self.selected_marker.item)
                    self.modified = True

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll',  1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll',  1, 'unit', event=event)

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        if self.__is_huge:  # image is huge and not totally in RAM
            band = bbox[3] - bbox[1]  # width of the tile band
            self.__tile[1][3] = band  # set the tile height
            self.__tile[2] = self.__offset + self.imagewidth * bbox[1] * 3  # set offset of the band
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imagewidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # image is totally in RAM
            return self.__pyramid[0].crop(bbox)

    def free_mem(self):
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()
        self.__permanentframe.destroy()

