from manim import*


SCALE_FACTOR = 1
tmp_pixel_height = config.pixel_height
config.pixel_height = config.pixel_width
config.pixel_width = tmp_pixel_height

config.frame_height = config.frame_height / SCALE_FACTOR
config.fram_width = config.frame_height * 9/16
FRAME_HEIGHT = config.frame_height
FRAME_WIDTH = config.frame_width


class PhoneBorder(Scene):
    def setup(self, add_border=True):
        if add_border:
            self.border = Rectangle(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            color = WHITE
        )
        self.add(self.border)


AXES_UNIT_SIZE = 2.5
RANGE = 3.1
 
class UnitCircle(Scene):
    CONFIG = {
        axes_config: {
            x_axis_config: {
                "x_min": -RANGE/2,
                "x_max": RANGE/2,
                "unit_size": AXES_UNIT_SIZE,
                "include_numbers": True
            },
            y_axis_config: {
                "label_direction": UP,
                "x_min": -RANGE/2,
                "x_max": RANGE/2,
                "unit_size": AXES_UNIT_SIZE,
                "include_numbers": True
            },
        },
    }
    def construct(self):
        self.axes   = axes   = self.get_axes()
        self.ghost_circle = ghost_circle = self.get_ghost_circle()
        self.radius = radius = self.get_radius(0)
        self.angle  = angle  = self.get_angle()
        self.h = h = self.get_h()
 
        circle_group = VGroup(radius, angle, h)
 
        def update_group(vg, alpha):
            theta = interpolate(0,4*PI,alpha) % (2*PI)
            r,a,h = vg
            r.become(self.get_radius(theta))
            a.become(self.get_angle(theta))
            h.become(self.get_h())
 
        self.add(axes,ghost_circle,radius,angle,h)
        self.play(
            UpdateFromAlphaFunc(
                circle_group,
                update_group,
                run_time=18,
                rate_func=linear,
            )
        )
        self.wait()
 
    def get_axes(self):
        axes = Axes(**self.axes_config)
        # FIX Y LABELS
        y_labels = axes.get_y_axis().numbers
        for label in y_labels:
            label.rotate(-PI/2)
        return axes
 
    def get_ghost_circle(self):
        return Circle(
            radius=self.axes.x_axis_config["unit_size"],
            color=GRAY,
            stroke_opacity=0.5
        )
 
    def get_radius(self,theta=0):
        return Line(
            self.ghost_circle.get_center(),
            self.ghost_circle.point_at_angle(theta),
            color=ORANGE
        )
 
    def get_angle(self,theta=0):
        return Arc(
            radius=self.ghost_circle.radius*0.18,
            arc_center=self.ghost_circle.get_center(),
            start_angle=0,
            angle=theta,
        )
 
    def get_h(self):
        return Line(
            self.radius.get_end(),
            [self.radius.get_end()[0],0,0]
        )
        
