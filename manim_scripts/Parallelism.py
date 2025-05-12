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
    min_x = float("inf")
    max_x = float("-inf")
    if isinstance(start, Group):
        for smo in start.submobjects:
            min_x = min(min_x, smo.get_bottom()[0])
            max_x = max(max_x, smo.get_bottom()[0])

    if isinstance(end, Group):
        for smo in end.submobjects:
            min_x = min(min_x, smo.get_bottom()[0])
            max_x = max(max_x, smo.get_bottom()[0])
    left = center.copy()
    left[0] = min_x
    right = center.copy()
    right[0] = max_x
    return left, right

def connect(v_line, obj, up=True, *args, **kwargs):
    if is_grp(obj):
        lines = []
        for smo in obj.submobjects: 
            start = smo.get_bottom() if up else smo.get_top()
            end = v_line.get_center().copy()
            end[0] = start[0]
            lines.append(Line(start, end, *args, **kwargs))
        return lines
    loc = obj.get_bottom() if up else obj.get_top()
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
            def __init__(self, text=None, formula=None, *args, **kwargs):
                super().__init__()
                self.block = Prism(*args, **kwargs)
                
                self.t = None
                self.f = None
                self.add(self.block)
                self.showing_text = True
                if text is not None:
                    self.t = Tex(text).move_to(self.block.get_corner(OUT))
                    self.add(self.t)
                if formula is not None:
                    self.f = Tex(formula).move_to(self.block.get_corner(OUT))
                    self.add(self.f)

            def create(self):
                anims = [ShowCreation(self.block)]
                if self.t:
                    anims.append(Write(self.t))
                return LaggedStart(*anims)

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
                        self.b_s.append(x.get_bottom())
                        self.b_e.append(x.get_top())
                    for x in self.top:
                        self.add(x)
                        self.t_s.append(x.get_bottom())
                        self.t_e.append(x.get_top())
                else:
                    self.is_grp = False
                    self.l = Line(start, end, *args, **kwargs)
                    self.add(self.l)
                    self.s = self.l.get_bottom()
                    self.e = self.l.get_top()
                self.args = args
                self.kwargs = kwargs
            
            def create(self):
                if self.is_grp:
                    return AnimationGroup(*[ShowCreation(x) for x in self.bot + [self.v_line] + self.top])
                return ShowCreation(self.l)

            def extend(self, dist):
                if self.is_grp:
                    anims = []
                    for i, x in enumerate(self.bot):
                        self.b_s[i] += dist*DOWN
                        anims.append(Transform(x, Line(self.b_s[i], self.b_e[i], *self.args, **self.kwargs)))

                    for i, x in enumerate(self.top):
                        self.t_e[i] += dist*UP
                        anims.append(Transform(x, Line(self.t_s[i], self.t_e[i], *self.args, **self.kwargs)))

                    return AnimationGroup(*anims)
                self.s += dist*DOWN
                self.e += dist*UP
                return Transform(self.l, Line(self.s, self.e, *self.args, **self.kwargs)) 

        class Residual(Group):
            def __init__(self, start, end, x, *args, **kwargs):
                super().__init__()
                self.start = start
                self.end = end
                self.s_point = start.get_left() + LEFT
                self.e_point = end.get_left() + LEFT
                self.e_point[0] = x
                self.s_point[0] = x

                self.s_line = Line(start.get_left(), self.s_point, *args, **kwargs)
                self.h_line = Connector(self.s_point, self.e_point, *args, **kwargs)
                self.e_line = Line(self.e_point, end.get_left(), *args, **kwargs)
                self.add(self.s_line)
                self.add(self.h_line)
                self.add(self.e_line)

            def create(self):
                return AnimationGroup(*[ShowCreation(x) for x  in self.submobjects])

            def extend(self, dist):
                return AnimationGroup(self.s_line.animate.shift(dist*DOWN),
                                      self.h_line.extend(dist),
                                      self.e_line.animate.shift(dist*UP))

        class TransformerBlock(Group):
            def __init__(self):
                super().__init__()
                self.std_width = 8
                self.std_height = 1.5
                
                self.rms_norm1 = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width, height=self.std_height)
                
                self.q_proj = FBlock("Q = XW_q", width=self.std_width/3, height=self.std_height)
                self.k_proj = FBlock("K = XW_k", width=self.std_width/3, height=self.std_height)
                self.v_proj = FBlock("V = XW_v", width=self.std_width/3, height=self.std_height)
                self.qkv_group = Group(self.q_proj, self.k_proj, self.v_proj)
                self.qkv_group.arrange(RIGHT, buff=0.5)

                self.attn = FBlock("Attention", "\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V",
                                   width=self.std_width, height=self.std_height)
                
                self.residual1 = FBlock("+", width=self.std_width//4, height=self.std_height)
                
                self.rms_norm2 = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width, height=self.std_height)
                
                self.ffn_gate = FBlock("XW_g", width=self.std_width/2, height=self.std_height)
                self.ffn_up = FBlock("XW_u", width=self.std_width/2, height=self.std_height)
                
                self.ffn_group = Group(self.ffn_gate, self.ffn_up)
                self.ffn_group.arrange(RIGHT, buff=0.5)
                
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

                self.arrange(UP, buff=1)

                lines = []
                for x1, x2 in zip(self.submobjects, self.submobjects[1:]):
                    l = Connector(x1, x2)
                    lines.append(l)
                i = len(self.submobjects) - 1
                for l in reversed(lines):
                    self.insert_submobject(i, l)
                    i -= 1

                res_x = self.rms_norm1.get_left()[0] - 1

                self.res = Residual(self.rms_norm1, self.residual1, res_x)
                self.add(self.res)
                self.res2 = Residual(self.submobjects[self.submobjects.index(self.residual1) + 1], self.residual2, res_x)
                self.add(self.res2)

                self.high_level = FBlock("Transformer Block", width = self.get_width() * 1.05, height = self.get_height() * 1.05)
                self.add(self.high_level)

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
                            anims.append(smo.animate.shift(dist*DOWN))
                        else:
                            anims.append(smo.animate.shift(dist*UP))
                        continue
                    if i <= idx:
                        anims.append(smo.animate.shift(dist*DOWN))
                    elif i == idx + 1:
                        assert isinstance(smo, Connector)
                        anims.append(smo.extend(dist))
                    else:
                        anims.append(smo.animate.shift(dist*UP))
                return AnimationGroup(*anims)


            def shrink_at(self, obj, dist=2):
                return self.extend_at(obj, dist=-dist)

            def create(self, high_level=True):
                self.is_hl = high_level
                if high_level:
                    return self.high_level.create()
                else:
                    anims = []
                    for obj in self:
                        if obj is self.high_level:
                            continue
                        if hasattr(obj, "create"):
                            anims.append(obj.create())
                        else:
                            anims.append(ShowCreation(obj))
                    return LaggedStart(*anims)

            def transform(self):
                if self.is_hl:
                    self.apply_depth_test()
                    self.high_level.deactivate_depth_test()
                    ret = AnimationGroup(FadeOut(self), self.high_level.create())
                    self.is_hl = False
                else:
                    self.deactivate_depth_test()
                    self.high_level.apply_depth_test()
                    ret = AnimationGroup(FadeOut(self.high_level), self.create())
                    self.is_hl = True
                return ret

        # x = FBlock("one").shift(2*UP)
        # y = Group(FBlock("two"), FBlock("Three")).arrange(RIGHT).shift(2*DOWN)
        # self.play(ShowCreation(x), ShowCreation(y))
        # conn = Connector(x, y)
        # self.play(conn.create())
        t = TransformerBlock()
        self.play(t.create())
        # self.play(t.extend_at(t.qkv_group))
        # self.play(t.transform())
        # self.play(t.transform())
