import os
from manimlib import *
from math import radians

# from manim_voiceover.services.gtts import GTTSService
# from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from voicover_gl import VoiceoverScene
import moderngl

def is_grp(obj):
    return obj.__class__ is Group

def get_vline_start_end(start, end):
    center = (start.get_center() + end.get_center()) / 2
    min_y = float("inf")
    max_y = float("-inf")
    if isinstance(start, Group):
        for smo in start.submobjects:
            max_y = max(max_y, smo.get_left()[1])
            min_y = min(min_y, smo.get_left()[1])

    if isinstance(end, Group):
        for smo in end.submobjects:
            min_y = min(min_y, smo.get_left()[1])
            max_y = max(max_y, smo.get_left()[1])
    left = center.copy()
    left[1] = min_y
    right = center.copy()
    right[1] = max_y
    return left, right

def connect(v_line, obj, up=True, *args, **kwargs):
    if is_grp(obj):
        lines = []
        for smo in obj.submobjects: 
            start = smo.get_left() if up else smo.get_right()
            end = v_line.get_center().copy()
            end[1] = start[1]
            lines.append(Line(start, end, *args, **kwargs))
        return lines
    loc = obj.get_left() if up else obj.get_right()
    return [Line(v_line.get_center(), loc, *args, **kwargs)]


class Parallelism(Scene):
    def construct(self):
        # self.set_speech_service(
        #     # RecorderService(transcription_model="base")
        #     GTTSService(transcription_model="base")
        #     )
        shader_dir = os.path.dirname(os.path.abspath(__file__)) + "/shaders/one_sided"
        Square3D.shader_folder = shader_dir

        class FBlock(Group):
            def __init__(self, text=None, formula=None, text_scale=1, *args, **kwargs):
                super().__init__()
                self.block = Prism(square_resolution=(10,10),*args, **kwargs)
                
                self.t = None
                self.f = None
                self.add(self.block)
                self.showing_text = True
                if text is not None:
                    self.t = Text(text).move_to(self.block.get_corner(OUT)).scale(text_scale)
                    self.add(self.t)
                if formula is not None:
                    self.f = Tex(formula).move_to(self.block.get_corner(OUT)).scale(text_scale)
                    self.add(self.f)

            def create(self, *args, **kwargs):
                anims = [ShowCreation(self.block, *args, **kwargs)]
                if self.t:
                    anims.append(Write(self.t, *args, **kwargs))
                return AnimationGroup(*anims)

            def transform(self):
                self.showing_text = not self.showing_text
                if self.showing_text:
                    return ReplacementTransform(self.f, self.t)
                return ReplacementTransform(self.t, self.f)

        class Connector(Group):
            def __init__(self, start, end, *args, **kwargs):
                super().__init__()
                if is_grp(start) or is_grp(end):
                    self.is_grp = True
                    l, r = get_vline_start_end(start, end)
                    self.v_line = Line(l, r, *args, **kwargs)
                    self.add(self.v_line)
                    self.bot = connect(self.v_line, start, False, *args, **kwargs)
                    self.top = connect(self.v_line, end, True, *args, **kwargs)
                    self.b_s = []
                    self.b_e = []
                    self.t_s = []
                    self.t_e = []
                    for x in self.bot:
                        self.add(x)
                        self.b_s.append(x.get_left())
                        self.b_e.append(x.get_right())
                    for x in self.top:
                        self.add(x)
                        self.t_s.append(x.get_left())
                        self.t_e.append(x.get_right())
                else:
                    self.is_grp = False
                    self.l = Line(start, end, *args, **kwargs)
                    self.add(self.l)
                    self.s = self.l.get_left()
                    self.e = self.l.get_right()
                self.args = args
                self.kwargs = kwargs
            
            def create(self, *args, **kwargs):
                if self.is_grp:
                    return AnimationGroup(*[ShowCreation(x, *args, **kwargs) for x in self.bot + [self.v_line] + self.top])
                return ShowCreation(self.l, *args, **kwargs)

            def extend(self, dist):
                if self.is_grp:
                    anims = []
                    for i, x in enumerate(self.bot):
                        self.b_s[i] += dist*LEFT
                        anims.append(Transform(x, Line(self.b_s[i], self.b_e[i], *self.args, **self.kwargs)))

                    for i, x in enumerate(self.top):
                        self.t_e[i] += dist*RIGHT
                        anims.append(Transform(x, Line(self.t_s[i], self.t_e[i], *self.args, **self.kwargs)))

                    return AnimationGroup(*anims)
                self.s += dist*LEFT
                self.e += dist*RIGHT
                return Transform(self.l, Line(self.s, self.e, *self.args, **self.kwargs)) 

        class Residual(Group):
            def __init__(self, start, end, y, *args, **kwargs):
                super().__init__()
                self.start = start
                self.end = end
                self.s_point = start.get_top() + UP
                self.e_point = end.get_top() + UP
                self.e_point[1] = y
                self.s_point[1] = y

                self.s_line = Line(start.get_top(), self.s_point, *args, **kwargs)
                self.h_line = Connector(self.s_point, self.e_point, *args, **kwargs)
                self.e_line = Line(self.e_point, end.get_top(), *args, **kwargs)
                self.add(self.s_line)
                self.add(self.h_line)
                self.add(self.e_line)

            def create(self, *args, **kwargs):
                return AnimationGroup(*[ShowCreation(x) for x  in self.submobjects])

            def extend(self, dist):
                return AnimationGroup(self.s_line.animate.shift(dist*LEFT),
                                      self.h_line.extend(dist),
                                      self.e_line.animate.shift(dist*RIGHT))

        class TransformerBlock(Group):
            def __init__(self, width, height):
                super().__init__()
                self.std_width = width 
                self.std_height = height 
                
                self.rms_norm1 = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width, height=self.std_height)
                
                self.q_proj = FBlock("Q = XW_q", width=self.std_width, height=self.std_height/3)
                self.k_proj = FBlock("K = XW_k", width=self.std_width, height=self.std_height/3)
                self.v_proj = FBlock("V = XW_v", width=self.std_width, height=self.std_height/3)
                self.qkv_group = Group(self.q_proj, self.k_proj, self.v_proj)
                self.qkv_group.arrange(DOWN, buff=0.5)

                self.attn = FBlock("Attention", "\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V",
                                   width=self.std_width, height=self.std_height)
                
                self.residual1 = FBlock("+", width=self.std_width//4, height=self.std_height)
                
                self.rms_norm2 = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width, height=self.std_height)
                
                self.ffn_gate = FBlock("XW_g", width=self.std_width, height=self.std_height/2)
                self.ffn_up = FBlock("XW_u", width=self.std_width, height=self.std_height/2)
                
                self.ffn_group = Group(self.ffn_gate, self.ffn_up)
                self.ffn_group.arrange(DOWN, buff=0.5)
                
                self.swiglu = FBlock("SwiGLU", r"x \cdot w \cdot \frac{1}{e^{-x}}",
                                     width=self.std_width, height=self.std_height)
                
                self.residual2 = FBlock("+", width=self.std_width//4, height=self.std_height)
                
                self.add(self.rms_norm1)
                self.add(self.qkv_group)
                self.add(self.attn)
                self.add(self.residual1)
                self.add(self.rms_norm2)
                self.add(self.ffn_group)
                self.add(self.swiglu)
                self.add(self.residual2)

                self.arrange(RIGHT, buff=1)

                lines = []
                for x1, x2 in zip(self.submobjects, self.submobjects[1:]):
                    l = Connector(x1, x2)
                    lines.append(l)
                i = len(self.submobjects) - 1
                for l in reversed(lines):
                    self.insert_submobject(i, l)
                    i -= 1

                res_y = self.rms_norm1.get_top()[1] + 1

                self.res = Residual(self.rms_norm1, self.residual1, res_y)
                self.add(self.res)
                self.res2 = Residual(self.submobjects[self.submobjects.index(self.residual1) + 1], self.residual2, res_y)
                self.add(self.res2)

                self.high_level = FBlock("Transformer\nBlock", width = self.get_width(), height = self.get_height(), text_scale=4)

            def extend_at(self, obj, dist=2):
                idx = self.submobjects.index(obj)
                anims = []
                for i, smo in enumerate(self):
                    if isinstance(smo, Residual):
                        s_idx = self.submobjects.index(smo.start)
                        e_idx = self.submobjects.index(smo.end)
                        if s_idx < idx and e_idx > idx:
                            anims.append(smo.extend(dist))
                        elif e_idx <= idx:
                            anims.append(smo.animate.shift(dist*LEFT))
                        else:
                            anims.append(smo.animate.shift(dist*RIGHT))
                        continue
                    if i <= idx:
                        anims.append(smo.animate.shift(dist*LEFT))
                    elif i == idx + 1:
                        assert isinstance(smo, Connector)
                        anims.append(smo.extend(dist))
                    else:
                        anims.append(smo.animate.shift(dist*RIGHT))
                return AnimationGroup(*anims)


            def shrink_at(self, obj, dist=2):
                return self.extend_at(obj, dist=-dist)

            def create(self, high_level=False, *args, **kwargs):
                self.is_hl = high_level
                if high_level:
                    self.high_level.move_to(self)
                    return self.high_level.create(*args, **kwargs)
                else:
                    anims = []
                    for obj in self:
                        if hasattr(obj, "create"):
                            anims.append(obj.create(*args, **kwargs))
                        else:
                            anims.append(ShowCreation(obj, *args, **kwargs))
                    return AnimationGroup(*anims)

            def transform(self):
                if self.is_hl:
                    self.deactivate_depth_test()
                    self.high_level.apply_depth_test()
                    ret = AnimationGroup(FadeOut(self.high_level), self.create(False))
                    self.is_hl = False
                else:
                    self.apply_depth_test()
                    self.high_level.deactivate_depth_test()
                    ret = AnimationGroup(FadeOut(self), self.create(True))
                    self.is_hl = True
                return ret

            def duplicate_to(self, target):
                anims = []
                if self.is_hl:
                    anims.append(ReplacementTransform(self.high_level.copy(), target.high_level.move_to(target)))
                else:
                    for i, x in self.submobjects:
                        anims.append(ReplacementTransform(x.copy(), target.submobjects[i]))
                return AnimationGroup(*anims)

        class Transformer(Group):
            def __init__(self, std_width=4, std_height=4, num_blocks=6, *args, **kwargs):
                super().__init__()

                self.std_width = std_width
                self.std_height = std_height

                self.transformer_layers = []
                for _ in range(num_blocks):
                    self.transformer_layers.append(TransformerBlock(self.std_width, self.std_height))

                self.embeddings = FBlock("Embedding", width=self.std_width*1.5,
                                         height=self.transformer_layers[-1].get_height(), text_scale=2)
                self.add(self.embeddings)

                for tb in self.transformer_layers:
                    self.add(tb)

                self.arrange(RIGHT, buff=3)

                lines = []
                for x1, x2 in zip(self.submobjects, self.submobjects[1:]):
                    l = Connector(x1, x2)
                    lines.append(l)
                i = len(self.submobjects) - 1
                for l in reversed(lines):
                    self.insert_submobject(i, l)
                    i -= 1

            def create(self, *args, **kwargs):
                anims = []
                for obj in self:
                    if hasattr(obj, "create"):
                        anims.append(obj.create(*args, **kwargs))
                    else:
                        anims.append(ShowCreation(obj, *args, **kwargs))
                return AnimationGroup(*anims)

            def duplicate_to(self, target):
                anims = []
                for i, x in enumerate(self.submobjects):
                    if hasattr(x, "duplicate_to"):
                        anims.append(x.duplicate_to(target.submobjects[i]))
                    else:
                        anims.append(ReplacementTransform(x.copy(), target.submobjects[i]))
                return AnimationGroup(*anims)


        t = TransformerBlock(4, 4)
        t2 = Transformer(4, 4).next_to(t, DOWN)
        
        def monkey_patch_interp(self, alpha, full=t):
            x_min = t.get_left()[0]
            x_max = t.get_right()[0]
            x_total = x_max - x_min
            m_x_min = self.mobject.get_left()[0]
            m_x_max = self.mobject.get_right()[0]
            m_x_total = m_x_max - m_x_min
            percentage = m_x_total/x_total

            x_curr = alpha*x_total
            x_start = (m_x_min - x_min) / x_total
            x_end = (m_x_max - x_min) / x_total
            new_alpha = (alpha - x_start)/(x_end - x_start + 1e-6)
            print(new_alpha, alpha, x_start, x_end, percentage)
            return self.interpolate_mobject(clip(new_alpha, 0, 1))

        def get_sub_alpha(
            self,
            alpha: float,
            index: int,
            num_submobjects: int
        ) -> float:
            # TODO, make this more understanable, and/or combine
            # its functionality with AnimationGroup's method
            # build_animations_with_timings
            # lag_ratio = self.lag_ratio
            # full_length = (num_submobjects - 1) * lag_ratio + 1
            # value = alpha * full_length
            # lower = index * lag_ratio
            # raw_sub_alpha = clip((value - lower), 0, 1)
            raw_sub_alpha = alpha


            x_min = t.get_left()[0]
            x_max = t.get_right()[0]
            x_total = x_max - x_min
            m_x_min = self.mobject.get_left()[0]
            m_x_max = self.mobject.get_right()[0]
            m_x_total = m_x_max - m_x_min

            x_start = (m_x_min - x_min) / x_total
            x_end = (m_x_max - x_min) / x_total
            new_alpha = (raw_sub_alpha - x_start)/(x_end - x_start + 1e-6)
            # print(alpha, new_alpha)

            return self.rate_func(new_alpha)

        def rgba_func(
        ) -> float:
            x_min = t.get_left()[0]
            x_max = t.get_right()[0]
            x_total = x_max - x_min
            m_x_min = self.mobject.get_left()[0]
            m_x_max = self.mobject.get_right()[0]
            m_x_total = m_x_max - m_x_min

            x_start = (m_x_min - x_min) / x_total
            x_end = (m_x_max - x_min) / x_total
            new_alpha = (raw_sub_alpha - x_start)/(x_end - x_start + 1e-6)
            # print(alpha, new_alpha)

            return self.rate_func(new_alpha)

        def updater(m, dt):
            camera_x = self.frame.get_center()[0]
            for mob in t.get_family(True):
                points = mob.get_points()
                if len(points):
                    try:
                        rgba = mob.data["rgba"].copy()
                        m_x_min = np.min(points[:, 0])
                        m_x_max = np.max(points[:, 0])
                        m_x_total = m_x_max - m_x_min
                        rgba[:, 3] = np.clip((camera_x-points[:, 0]), 0, 1)
                        mob.set_rgba_array(rgba)
                    except:
                        pass
                

        # self.play(t.create(), self.frame.animate.match_width(t))
        # Animation.get_sub_alpha = get_sub_alpha
        b = FBlock(width=8).next_to(t, DOWN)
        b2 = FBlock(width=8).next_to(b, RIGHT)
        # self.play(t.create(), run_time=10)

        # self.play(t.create(), self.frame.animate.match_width(t))
        # self.frame.save_state()
        # #TODO get camera to move to start
        self.play(t.create(), run_time=0)
        t.set_opacity(0)
        self.frame.add_updater(updater)
        self.play(self.frame.animate.shift(RIGHT * t.get_width()), run_time=10)
        # self.play(Restore(self.frame, run_time=2))
        # self.play(t.duplicate_to(t2))
