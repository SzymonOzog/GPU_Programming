import os
from manimlib import *
from math import radians

from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene
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
    else:
        max_y = max(max_y, start.get_left()[1])
        min_y = min(min_y, end.get_left()[1])

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
            lines.append(Line3D(start, end, *args, **kwargs))
        return lines
    loc = obj.get_left() if up else obj.get_right()
    start = v_line.get_center()
    start[1] = loc[1] 
    return [Line3D(start, loc, *args, **kwargs)]

mats = []


class Parallelism(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            # RecorderService(transcription_model="base")
            GTTSService(transcription_model="base")
            )
        shader_dir = os.path.dirname(os.path.abspath(__file__)) + "/shaders/one_sided"
        Square3D.shader_folder = shader_dir

        class FBlock(Group):
            def __init__(self, text=None, formula=None, text_scale=1, text_rotation_deg=-90, weights=None, *args, **kwargs):
                super().__init__()
                self.block = Prism(square_resolution=(10,10),*args, **kwargs)
                
                self.t = None
                self.f = None
                self.add(self.block)
                self.showing_text = True
                if text is not None:
                    self.t = Text(text).move_to(self.block.get_corner(OUT)).rotate(radians(text_rotation_deg)).scale(text_scale)
                    t_h = self.t.get_height()
                    t_w = self.t.get_width()
                    b_h = self.block.get_height()
                    b_w = self.block.get_width()
                    if b_h/t_h > b_w/t_w:
                        self.t.match_width(self.block)
                        self.t.rescale_to_fit(b_w*0.8, dim=0)
                        # self.t.match_x(
                    else:
                        # self.t.match_height(self.block)
                        self.t.rescale_to_fit(b_h*0.8, dim=1)
                    self.add(self.t)
                if formula is not None:
                    self.f = Tex(formula).move_to(self.block.get_corner(OUT)).scale(text_scale)

            def set_weights(self, weights):
                self.w = weights
                self.w_end = self.w.copy().move_to(self.block.get_center()).scale(0.01)
                mats.append((self.w.copy(), self.w, self.w_end, self))


            def create(self, *args, **kwargs):
                anims = [ShowCreation(self.block, *args, **kwargs)]
                if self.t:
                    anims.append(Write(self.t, *args, **kwargs))
                return AnimationGroup(*anims)

            def transform(self):
                self.showing_text = not self.showing_text
                if self.showing_text:
                    self.add(self.t)
                    self.remove(self.f)
                    return ReplacementTransform(self.f, self.t)
                self.add(self.f)
                self.remove(self.t)
                return ReplacementTransform(self.t, self.f)

        class Connector(Group):
            def __init__(self, start, end, *args, **kwargs):
                super().__init__()
                self.v_line = None
                if is_grp(start):
                    self.is_grp = True
                    self.bot = []
                    self.top = []
                    self.b_s = []
                    self.b_e = []
                    for s in start:
                        st = s.get_right()
                        en = end.get_left().copy()
                        en[1] = st[1]
                        l = Line3D(st, en, *args, **kwargs)
                        self.bot.append(l)
                        self.add(l)
                        self.b_s.append(l.get_left())
                        self.b_e.append(l.get_right())
                elif is_grp(end):
                    self.is_grp = True
                    l, r = get_vline_start_end(start, end)
                    self.v_line = Line3D(l, r, *args, **kwargs)
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
                    self.s = start if isinstance(start, np.ndarray) else start.get_right()
                    self.e = end if isinstance(end, np.ndarray) else end.get_left()
                    self.l = Line3D(self.s, self.e, *args, **kwargs)
                    self.add(self.l)
                self.args = args
                self.kwargs = kwargs
            
            def create(self, *args, **kwargs):
                if self.is_grp:
                    lines = self.bot + self.top
                    if self.v_line is not None:
                        lines.append(self.v_line)
                    return AnimationGroup(*[ShowCreation(x, *args, **kwargs) for x in lines])
                return ShowCreation(self.l, *args, **kwargs)

            def extend(self, dist):
                if self.is_grp:
                    anims = []
                    for i, x in enumerate(self.bot):
                        self.b_s[i] += dist*LEFT
                        anims.append(Transform(x, Line3D(self.b_s[i], self.b_e[i], *self.args, **self.kwargs)))

                    for i, x in enumerate(self.top):
                        self.t_e[i] += dist*RIGHT
                        anims.append(Transform(x, Line3D(self.t_s[i], self.t_e[i], *self.args, **self.kwargs)))

                    return AnimationGroup(*anims)
                self.s += dist*LEFT
                self.e += dist*RIGHT
                return Transform(self.l, Line3D(self.s, self.e, *self.args, **self.kwargs)) 

        class Residual(Group):
            def __init__(self, start, end, y, *args, **kwargs):
                super().__init__()
                self.start = start
                self.end = end
                self.s_point = start.get_top() + UP
                self.e_point = end.get_top() + UP
                self.e_point[1] = y
                self.s_point[1] = y

                self.s_line = Line3D(start.get_top(), self.s_point, *args, **kwargs)
                self.h_line = Connector(self.s_point, self.e_point, *args, **kwargs)
                self.e_line = Line3D(self.e_point, end.get_top(), *args, **kwargs)
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
            def __init__(self, width, height, *args, **kwargs):
                super().__init__()
                self.std_width = width 
                self.std_height = height 
                
                self.rms_norm1 = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width*2/3, height=self.std_height, color=YELLOW_E)
                
                self.rotary1 = FBlock("RoPE", width=self.std_width/3, height=self.std_height/3, *args, **kwargs)
                self.rotary2 = FBlock("RoPE", width=self.std_width/3, height=self.std_height/3, *args, **kwargs)

                self.q_proj = FBlock("Q Proj","Q = XW_q", width=self.std_width*2/3, height=self.std_height/3, color=TEAL, *args, **kwargs)
                self.k_proj = FBlock("K Proj","K = XW_k", width=self.std_width*2/3, height=self.std_height/3, color=TEAL, *args, **kwargs)
                self.v_proj = FBlock("V Proj","V = XW_v", width=self.std_width*2/3, height=self.std_height/3, color=TEAL, *args, **kwargs)
                self.qkv_group = Group(
                        Group(self.q_proj, self.rotary1).arrange(RIGHT, buff=0.4).add(Connector(self.q_proj, self.rotary1, width=0.1, color=WHITE)) , 
                        Group(self.k_proj, self.rotary2).arrange(RIGHT, buff=0.4).add(Connector(self.k_proj, self.rotary2, width=0.1, color=WHITE)) , 
                        self.v_proj)
                self.qkv_group.arrange(DOWN, buff=0.5, aligned_edge=LEFT)

                self.attn = FBlock("Attention", "\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V",
                                   width=self.std_width, height=self.std_height, color=GOLD_E, *args, **kwargs)

                self.out_proj = FBlock("Out Proj","Y = XW_v", width=self.std_width, height=self.std_height, color=TEAL, *args, **kwargs)
                
                self.residual1 = FBlock("+", width=self.std_width//4, height=self.std_height//4)
                
                self.rms_norm2 = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width*2/3, height=self.std_height, color=YELLOW_E)
                
                self.ffn_gate = FBlock("Gate", "XW_g", width=self.std_width, height=self.std_height/2, color=TEAL, *args, **kwargs)
                self.ffn_up = FBlock("Up Proj", "XW_u", width=self.std_width, height=self.std_height/2, color=TEAL, *args, **kwargs)
                
                self.ffn_group = Group(self.ffn_gate, self.ffn_up)
                self.ffn_group.arrange(DOWN, buff=0.5)
                
                self.swiglu = FBlock("SwiGLU", r"x \cdot w \cdot \frac{1}{e^{-x}}",
                                     width=self.std_width/2, height=self.std_height)
                
                self.ffn_down =  FBlock("Down Proj", "XW_d", width=self.std_width, height=self.std_height, color=TEAL, *args, **kwargs)
                self.residual2 = FBlock("+", width=self.std_width//4, height=self.std_height//4, *args, **kwargs)
                
                self.add(self.rms_norm1)
                self.add(self.qkv_group)
                self.add(self.attn)
                self.add(self.out_proj)
                self.add(self.residual1)
                self.add(self.rms_norm2)
                self.add(self.ffn_group)
                self.add(self.swiglu)
                self.add(self.ffn_down)
                self.add(self.residual2)

                self.arrange(RIGHT, buff=1.5)

                lines = []
                for x1, x2 in zip(self.submobjects, self.submobjects[1:]):
                    l = Connector(x1, x2, width=0.1, color=WHITE)
                    lines.append(l)
                i = len(self.submobjects) - 1
                for l in reversed(lines):
                    self.insert_submobject(i, l)
                    i -= 1
                self.high_level = FBlock("Transformer\nBlock", text_rotation_deg=0, width = self.get_width(), height = self.get_height(), text_scale=4)

            def set_mats(self):
                mats = [Group(*[TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,h}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,h}"]]) 
                                for _ in range(4)]).arrange(OUT, buff=0.5).rotate(radians(25), DOWN).scale(0.5) for _ in range(3)]
                Group(*mats).arrange(DOWN).move_to(self.k_proj).shift(2*RIGHT)
                self.q_proj.set_weights(mats[0])
                self.k_proj.set_weights(mats[1])
                self.v_proj.set_weights(mats[2])

                mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,m}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,m}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{h,0}", "w_{h,1}", "\\cdots", "w_{h,m}"]]).rotate(radians(25), DOWN).scale(0.9)
                mat.move_to(self.out_proj).shift(2*RIGHT)
                self.out_proj.set_weights(mat)

                mats = [TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,h}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{i,0}", "w_{i,1}", "\\cdots", "w_{i,h}"]]).rotate(radians(25), DOWN).scale(0.7) for _ in range(2)]
                Group(*mats).arrange(DOWN).move_to(self.ffn_group).shift(2*RIGHT)
                self.ffn_gate.set_weights(mats[0])
                self.ffn_up.set_weights(mats[1])

                mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,i}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,i}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,i}"]]).rotate(radians(25), DOWN).scale(0.9)
                mat.move_to(self.ffn_down).shift(2*RIGHT)
                self.ffn_down.set_weights(mat)


            def create_residuals(self, inp):
                res_y = self.rms_norm1.get_top()[1] + 1
                self.res = Residual(inp, self.residual1, res_y, width=0.1, color=WHITE)
                self.add(self.res)
                self.res2 = Residual(self.submobjects[self.submobjects.index(self.residual1) + 1], self.residual2, res_y, width=0.1, color=WHITE)
                self.add(self.res2)


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

            def duplicate_to(self, target, copy=True):
                anims = []
                if self.is_hl:
                    anims.append(ReplacementTransform(self.high_level.copy() if copy else self.high_level, target.high_level.move_to(target)))
                else:
                    for i, x in enumerate(self.submobjects):
                        anims.append(ReplacementTransform(x.copy() if copy else x, target.submobjects[i]))
                return AnimationGroup(*anims)

        class Transformer(Group):
            def __init__(self, std_width=4, std_height=4, num_blocks=4, high_level=True, *args, **kwargs):
                super().__init__()

                self.std_width = std_width
                self.std_height = std_height

                self.transformer_layers = []
                self.high_levels = []
                for _ in range(num_blocks):
                    self.transformer_layers.append(TransformerBlock(self.std_width, self.std_height, *args, **kwargs))
                    self.high_levels.append(self.transformer_layers[-1].high_level)
                    self.transformer_layers[-1].is_hl = high_level

                self.embeddings = FBlock("Embedding", width=self.std_width*1.5,
                                         height=self.transformer_layers[-1].get_height(), text_scale=2, color=RED, *args, **kwargs)
                self.add(self.embeddings)

                if high_level:
                    for tb in self.high_levels:
                        self.add(tb)
                else:
                    for tb in self.transformer_layers:
                        self.add(tb)

                self.rms_norm = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width*2/3, height=self.std_height, color=YELLOW_E, *args, **kwargs)
                self.linear = FBlock("Linear", "XW", width=self.std_width, height=self.std_height, color=TEAL, *args, **kwargs)
                self.softmax = FBlock("Softmax", 
                                      r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}",
                                      width=self.std_width*2/3, height=self.std_height, color=GREEN, *args, **kwargs)
                self.add(self.rms_norm)
                self.add(self.linear)
                self.add(self.softmax)

                self.arrange(RIGHT, buff=2)

                self.lines = []
                for x1, x2 in zip(self.submobjects, self.submobjects[1:]):
                    l = Connector(x1, x2, width=0.1, color=WHITE)
                    self.lines.append(l)

                for l, tb, hl in zip(self.lines, self.transformer_layers, self.high_levels):
                    if high_level:
                        tb.move_to(hl)
                    tb.create_residuals(l)
                    tb.set_mats()
                i = len(self.submobjects) - 1
                for l in reversed(self.lines):
                    self.insert_submobject(i, l)
                    i -= 1

                mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,n}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,n}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,n}"]]).rotate(radians(25), DOWN).scale(0.7)
                mat.move_to(self.linear)
                self.linear.set_weights(mat)

            def create(self, *args, **kwargs):
                anims = []
                for obj in self:
                    if hasattr(obj, "create"):
                        anims.append(obj.create(*args, **kwargs))
                    else:
                        anims.append(ShowCreation(obj, *args, **kwargs))
                return AnimationGroup(*anims)

            def pipeline_parallelize(self, n, dist=8):
                split_blocks = []
                anims = []
                cum_dist = 0
                for i, b in enumerate(self.transformer_layers):
                    if i % (len(self.transformer_layers)/n) == 0 and i != 0:
                        cum_dist += dist
                    if b.is_hl:
                        anims.append(self.high_levels[i].animate.shift(cum_dist*DOWN))
                    else:
                        anims.append(b.animate.shift(cum_dist*DOWN))

                cum_dist = 0
                for i, b in enumerate(self.lines):
                    if i % (len(self.transformer_layers)/n) == 0 and i != 0 and i < len(self.transformer_layers):
                        tmp = Dot().move_to(b.get_left() + cum_dist*DOWN)
                        cum_dist += dist
                        tmp2 = Dot().move_to(b.get_right() + cum_dist*DOWN)
                        #TODO this is too much hacks
                        anims.append(Transform(b, Connector(tmp, Group(tmp2), width=0.1, color=WHITE)))
                    else:
                        anims.append(b.animate.shift(cum_dist*DOWN))
                anims.append(self.rms_norm.animate.shift(cum_dist*DOWN))
                anims.append(self.linear.animate.shift(cum_dist*DOWN))
                anims.append(self.softmax.animate.shift(cum_dist*DOWN))
                return AnimationGroup(*anims)

            def transform(self, indices=[]):
                anims = []
                for i in indices:
                    block = self.transformer_layers[i]
                    high_level = self.high_levels[i]
                    if block.is_hl:
                        block.move_to(high_level)
                        idx = self.submobjects.index(high_level)
                        self.remove(high_level)
                        self.insert_submobject(idx, block)
                    else:
                        high_level.move_to(block)
                        idx = self.submobjects.index(block)
                        self.remove(block)
                        self.insert_submobject(idx, high_level)
                    anims.append(block.transform())

                return AnimationGroup(*anims)

            def duplicate_to(self, target, copy=True):
                anims = []
                for i, x in enumerate(self.submobjects):
                    if hasattr(x, "duplicate_to"):
                        anims.append(x.duplicate_to(target.submobjects[i], copy))
                    else:
                        anims.append(ReplacementTransform(x.copy() if copy else x, target.submobjects[i]))
                return AnimationGroup(*anims)


        # t2 = Transformer(4, 4).next_to(t, DOWN)
        
        def updater(m, dt):
            camera_x = self.frame.get_center()[0]
            speed = 0.2
            for mob in t.get_family(True):
                points = mob.get_points()
                if len(points):
                    if "rgba" in mob.data_dtype.names:
                        rgba = mob.data["rgba"].copy()
                        m_x_min = np.min(points[:, 0])
                        m_x_max = np.max(points[:, 0])
                        m_x_total = m_x_max - m_x_min
                        rgba[:, 3] = -np.clip(((camera_x-points[:, 0])*speed), 0, 1)
                        mob.set_rgba_array(rgba)
                    else:
                        rgba_s = mob.data["stroke_rgba"].copy()
                        rgba_f = mob.data["fill_rgba"].copy()
                        m_x_min = np.min(points[:, 0])
                        m_x_max = np.max(points[:, 0])
                        m_x_total = m_x_max - m_x_min
                        rgba_f[:, 3] = np.clip(((camera_x-points[:, 0])*speed), 0, 1)
                        rgba_s[:, 3] = np.clip(((camera_x-points[:, 0])*speed), 0, 1)
                        mob.set_rgba_array(rgba_f, name="fill_rgba")
                        mob.set_rgba_array(rgba_s, name="stroke_rgba")
            for s, c, e, b in mats:
                # m_x_min = s.get_left()[0] - MAT_START_OFFSET - 1
                # m_x_max = s.get_right()[0] - MAT_START_OFFSET
                m_x_min = b.get_left()[0]
                m_x_max = b.get_center()[0]
                MAT_START_OFFSET = (s.get_left()[0] - m_x_min)/speed
                alpha = clip(camera_x - m_x_min, 0, 1)
                alpha2 = clip((camera_x - m_x_max)*speed, 0, 1)
                for m_s, m_c, m_e in zip(s.get_family(True), c.get_family(True), e.get_family(True)):
                    m_c = m_c.interpolate(m_s, m_e, alpha2)
                    if alpha <= 1:
                        points = m_c.get_points()
                        if len(points):
                            rgba_s = m_c.data["stroke_rgba"].copy()
                            rgba_f = m_c.data["fill_rgba"].copy()
                            rgba_f[:, 3] = np.clip(((camera_x-points[:, 0]+MAT_START_OFFSET)*speed), 0, 1)
                            rgba_s[:, 3] = np.clip(((camera_x-points[:, 0]+MAT_START_OFFSET)*speed), 0, 1)
                            m_c.set_rgba_array(rgba_f, name="fill_rgba")

        # This is so hacky I feel stupid
        def run_transformers(transformers, run_time=1, anim=None):
            global saved_colors, saved_x
            saved_colors = {}
            saved_x = {}
            if anim is not None:
                run_time = 0
                for a in anim:
                    run_time += a.run_time
            for t in transformers:
                saved_x[t] = t.get_left()[0]
            def flash_updater(m, dt):
                global saved_colors, saved_x
                for t in transformers:
                    change = (t.get_width())/run_time
                    saved_x[t] += change*dt
                    flash_x = saved_x[t]
                    for mob in t.get_family(True):
                        points = mob.get_points()
                        if len(points):
                            if "rgba" in mob.data_dtype.names:
                                if mob not in saved_colors:
                                    saved_colors[mob] = color_to_rgb(mob.get_color())
                                rgba = mob.data["rgba"].copy()
                                m_x_min = np.min(points[:, 0])
                                m_x_max = np.max(points[:, 0])
                                m_x_total = m_x_max - m_x_min
                                start = np.stack((color_to_rgb(YELLOW),)*len(points))
                                end = np.stack((saved_colors[mob],)*len(points))
                                alpha = np.clip(np.abs(flash_x - points[:, 0]), 0, 1)
                                new_color = start*(1 - alpha)[..., np.newaxis] + end*alpha[..., np.newaxis]
                                rgba[:, :3] = new_color
                                mob.set_rgba_array(rgba)
            self.frame.add_updater(flash_updater)
            if anim is None:
                self.wait(run_time)
            else:
                for a in anim:
                    self.play(a)
            self.wait(0.1)
            self.frame.remove_updater(flash_updater)
                
        # t.set_opacity(0)
        # for s,c,e,b in mats:
        #     self.add(c)
        # # self.add(t.mat)
        # self.frame.save_state()
        # # self.frame.set_euler_angles(-1.55674497,  0.5891779 ,  1.55853628).set_shape(30.301018, 17.031006).move_to([-85.44072  ,   1.0943325,  -0.4649295])
        # self.frame.set_euler_angles(-1.62773323,  0.46361119,  1.62378591).set_shape(28.885307, 16.235258).move_to([-92.19126   ,   0.4578367 ,   0.18124883])        
        # #animate creation
        # t.set_opacity(0)
        # self.wait(1)
        # self.frame.add_updater(updater)
        # self.play(self.frame.animate.shift(RIGHT * t.get_width() * 1.05), run_time=20, rate_func=linear)
        # self.frame.remove_updater(updater)
        # self.play(Restore(self.frame), AnimationGroup(*[x.transform() for x in t.transformer_layers]), run_time=2)
        # self.wait(1)
        # self.play(t.duplicate_to(t2))

        #Create the transformer
        transformer = Transformer(4, 12).shift(5*DOWN)
        self.play(transformer.create(), self.frame.animate.match_width(transformer))


        gpu0 = SurroundingRectangle(transformer, buff=2, color=GREEN)
        gpu0_t = Text("GPU0").set_color(GREEN).scale(10).next_to(gpu0, UP, aligned_edge=LEFT, buff=2)
        # Create GPU
        with self.voiceover(text="""In the simple case we have our model that is living on the GPU""") as trk:
            self.play(ShowCreation(gpu0))
            self.play(self.frame.animate.rescale_to_fit(gpu0.get_width() + 10, dim=0), Write(gpu0_t))

        cpu0_i = SVGMobject("./icons/cpu.svg").scale(8).set_color(WHITE).next_to(transformer, UP).shift(10*UP).set_color(BLUE)
        cpu0 = SurroundingRectangle(cpu0_i, buff=2, color=BLUE)
        cpu0_t = Text("CPU").set_color(BLUE).scale(10).next_to(cpu0, UP, aligned_edge=LEFT, buff=2)
        # Create CPU
        with self.voiceover(text="""And a CPU that is orchestrating it""") as trk:
            self.play(ShowCreation(cpu0_i), ShowCreation(cpu0), Write(cpu0_t))

        #run transformer
        with self.voiceover(text="""In a very simplified form the CPU is sending a task to process to the model,
                            getting the output, processing it and sending another task to our model living on the GPU""") as trk:
            while trk.get_remaining_duration() > 0:
                request = Square3D(color=RED, side_length=6).move_to(transformer.embeddings.get_left())
                self.play(FadeIn(request, shift=request.get_center() - cpu0_i.get_center(), remover=True), run_time=2)
                run_transformers([transformer])
                request = Square3D(color=RED, side_length=6).move_to(cpu0_i)
                self.play(FadeIn(request, shift=request.get_center() - transformer.softmax.get_right(), remover=True), run_time=2)


        # Create next GPU
        transformer2 = Transformer(4, 12).next_to(transformer, DOWN, buff=16)
        gpu1 = SurroundingRectangle(transformer2, buff=2, color=GREEN)
        gpu1_t = Text("GPU1").set_color(GREEN).scale(10).next_to(gpu1, UP, aligned_edge=LEFT, buff=2)

        with self.voiceover(text="""But let's say we get another GPU and want to use this fact to speed up our inference""") as trk:
            self.play(ShowCreation(gpu1))
            self.play(Write(gpu1_t))


        # Data parallel
        with self.voiceover(text="""One simple thing to do would be to just make an entire copy of our model and put it on the second GPU""") as trk:
            self.play(transformer.duplicate_to(transformer2))

        # Name it
        dp_t = Text("Data Parallel").scale(50).next_to(cpu0, UP, buff=8).set_color(RED)
        with self.voiceover(text="""This is called data parallelizm""") as trk:
            self.play(Write(dp_t))

        # run both
        with self.voiceover(text="""In this case, each GPU can process a different request, this essentially doubles our throughput. 
                            But there are a couple of issues with it""") as trk:
            while trk.get_remaining_duration() > 0:
                transformers = [transformer, transformer2]
                start_anims = []
                end_anims = []
                for t in transformers:
                    request_s = Square3D(color=RED, side_length=6).move_to(t.embeddings.get_left())
                    start_anims.append(FadeIn(request_s, shift=request_s.get_center() - cpu0_i.get_center(), remover=True))
                    request_e = Square3D(color=RED, side_length=6).move_to(cpu0_i)
                    end_anims.append(FadeIn(request_e, shift=request_e.get_center() - t.softmax.get_right(), remover=True))

                self.play(*start_anims, run_time=2)
                run_transformers(transformers)
                self.play(*end_anims, run_time=2)

        # Insufficient load
        with self.voiceover(text="""First of all, we might not have enough load, in this case the CPU is unable to 
                            schedule a task on both GPUs at once meaning that essentially one of those becomes idle""") as trk:
            while trk.get_remaining_duration() > 0:
                transformers = [transformer, transformer2]
                for t in transformers:
                    request = Square3D(color=RED, side_length=6).move_to(t.embeddings.get_left())
                    self.play(FadeIn(request, shift=request.get_center() - cpu0_i.get_center(), remover=True), run_time=2)
                    run_transformers([t])
                    request = Square3D(color=RED, side_length=6).move_to(cpu0_i)
                    self.play(FadeIn(request, shift=request.get_center() - t.softmax.get_right(), remover=True), run_time=2)

        
        with self.voiceover(text="""But the more popular reason for ditching data parallel is just the simple fact that the models of today
                            no longer fit on a single GPU""") as trk:
            pass

        # Make pp scene part
        cpu1 = cpu0.copy()
        cpu1_t = cpu0_t.copy()
        cpu1_i = cpu0_i.copy()
        transformer3 = Transformer(4, 12).move_to(transformer)
        gpu2 = gpu0.copy()
        gpu2_t = gpu0_t.copy()
        gpu3 = gpu1.copy()
        gpu3_t = gpu1_t.copy()
        cp = Group(cpu1, cpu1_t, cpu1_i, transformer3, gpu2, gpu2_t, gpu3, gpu3_t).next_to(
                Group(cpu0_t, gpu1), RIGHT, buff = 20)
        self.play(FadeIn(cp))

        # Make tp scene part
        cpu2 = cpu0.copy()
        cpu2_t = cpu0_t.copy()
        cpu2_i = cpu0_i.copy()
        transformer4 = Transformer(4, 12).move_to(transformer)
        gpu4 = gpu0.copy()
        gpu4_t = gpu0_t.copy()
        gpu5 = gpu1.copy()
        gpu5_t = gpu1_t.copy()
        cp2 = Group(cpu2, cpu2_t, cpu2_i, transformer4, gpu4, gpu4_t, gpu5, gpu5_t).next_to(
                Group(cpu1_t, gpu3), RIGHT, buff = 20)
        self.play(FadeIn(cp2))


        with self.voiceover(text="""This has created a needs for methods that split our model parameters across multiple GPUs""") as trk:
            self.play(self.frame.animate.shift(gpu2.get_center() - gpu0.get_center()), run_time=trk.get_remaining_duration())

        # Pipeline parallel
        pp_t = Text("Pipeline Parallel").scale(50).next_to(cpu1, UP, buff=5).set_color(YELLOW)
        dist = gpu2.get_center()[1] - gpu3.get_center() [1]
        with self.voiceover(text="""One such method would be Pipeline Parallelizm. In this setting, in this setting, a subset of 
                            our model layers is placed on one GPU  while the rest is placed on a different one""") as trk:
            self.play(Write(pp_t))
            self.play(transformer3.pipeline_parallelize(2, dist))


        with self.voiceover(text="""This effectively splits our model between two GPUs but introduces a few drawbacks""") as trk:
            while trk.get_remaining_duration() > 0:
                request = Square3D(color=RED, side_length=6).move_to(transformer3.embeddings.get_left())
                self.play(FadeIn(request, shift=request.get_center() - cpu1_i.get_center(), remover=True), run_time=2)
                a1 = AnimationGroup(gpu2.animate.set_color(GREEN), gpu3.animate.set_color(GREY))
                a2 = AnimationGroup(gpu2.animate.set_color(GREY), gpu3.animate.set_color(GREEN))
                a3 = AnimationGroup(gpu2.animate.set_color(GREY), gpu3.animate.set_color(GREY))
                run_transformers([transformer], 1, [a1, a2, a3])
                request = Square3D(color=RED, side_length=6).move_to(cpu1_i)
                self.play(FadeIn(request, shift=request.get_center() - transformer3.softmax.get_right(), remover=True), run_time=2)

        with self.voiceover(text="""The most important of which is simillar to the issue we had in data parallel. 
                            When we don't have enough running requests, only one of our GPU's is doing the work, while the other one is idling""") as trk:
            while trk.get_remaining_duration() > 0:
                request = Square3D(color=RED, side_length=6).move_to(transformer3.embeddings.get_left())
                self.play(FadeIn(request, shift=request.get_center() - cpu1_i.get_center(), remover=True), run_time=2)
                a1 = AnimationGroup(gpu2.animate.set_color(GREEN), gpu3.animate.set_color(GREY), run_time=2)
                a2 = AnimationGroup(gpu2.animate.set_color(GREY), gpu3.animate.set_color(GREEN), run_time=2)
                a3 = AnimationGroup(gpu2.animate.set_color(GREY), gpu3.animate.set_color(GREY), run_time=2)
                run_transformers([transformer], anim=[a1, a2, a3])
                request = Square3D(color=RED, side_length=6).move_to(cpu1_i)
                self.play(FadeIn(request, shift=request.get_center() - transformer3.softmax.get_right(), remover=True), run_time=2)
        # start TP
        tp_t = Text("Tensor Parallel").scale(50).next_to(cpu2, UP, buff=5).set_color(GREEN)
        with self.voiceover(text="""This has lead to a new method called Tensor Parallelizm""") as trk:
            self.play(self.frame.animate.shift(gpu4.get_center() - gpu2.get_center()), run_time=trk.get_remaining_duration())
            self.play(Write(tp_t))
        transformer5 = Transformer(4, 12).move_to(gpu5)
        for mob in it.chain(transformer5.get_family(True), *[x.get_family(True) for x in transformer5.transformer_layers]):
            if isinstance(mob, Prism):
                mob.set_color(GREY)
        with self.voiceover(text="""In tensor parallelizm, we create a copy of an entire model on the second GPU""") as trk:
            self.play(transformer4.duplicate_to(transformer5))
            self.play(transformer4.transform([0, 1, 2, 3]), transformer5.transform([0, 1, 2, 3]))


        transformer6 = Transformer(4,4, high_level=False, text_rotation_deg=0).move_to(transformer4, aligned_edge=DOWN)
        transformer7 = Transformer(4,4, high_level=False, text_rotation_deg=0).move_to(transformer5, aligned_edge=UP)
        for mob in it.chain(transformer7.get_family(True), *[x.get_family(True) for x in transformer7.transformer_layers]):
            if isinstance(mob, Prism):
                mob.set_color(GREY)
        with self.voiceover(text="""And we split all model weights and calculations across all stages of the model""") as trk:
            self.play(transformer4.duplicate_to(transformer6, copy=False), transformer5.duplicate_to(transformer7, copy=False))

        def split_weights(t1_mobs, t2_mobs, color, dim=0, run_time=1):
            anims = []
            for b, b2 in zip(t1_mobs, t2_mobs):

                down = b.copy().set_height(b.get_height()/2, True)
                down.move_to(b, aligned_edge=DOWN).set_color(color).scale(1.01)

                up = b.copy().set_height(b.get_height()/2, True)
                up.move_to(b2, aligned_edge=DOWN).set_color(color).scale(1.01)
                anims.append(Transform(down, up, remover=True))

                mid = b.get_center()[dim]
                self.add(down)
                self.add(up)
                self.remove(up)
                for smo in b.block.submobjects:
                    points = smo.get_points()
                    if len(points):
                        rgba = smo.data["rgba"].copy()
                        if dim == 0:
                            rgba[points[:, dim] > mid, :] = color_to_rgba(GREY)
                        else:
                            rgba[points[:, dim] < mid, :] = color_to_rgba(GREY)
                        smo.set_rgba_array(rgba)
            # some hacky way to fix manim z ordering
            for b, b2 in zip(t1_mobs, t2_mobs):
                self.add(b.t)
                self.add(b2.t)

            self.play(*anims, run_time=run_time)

            for b, b2 in zip(t1_mobs, t2_mobs):
                mid = b2.get_center()[dim]
                for smo in b2.block.submobjects:
                    points = smo.get_points()
                    if len(points):
                        rgba = smo.data["rgba"].copy()
                        if dim == 0:
                            rgba[points[:, dim] > mid, :] = color_to_rgba(color)
                        else:
                            rgba[points[:, dim] < mid, :] = color_to_rgba(color)
                        smo.set_rgba_array(rgba)

        with self.voiceover(text="""The question becomes, how to split the weights across the model""") as trk:
            pass

        focus_obj = Group(transformer6.embeddings, transformer7.embeddings)

        with self.voiceover(text="""Going layer by layer, we first encounter our embeddings. The way we split them is that we just
                            divide them equally across all GPUs, each embedding only the tokens it has in it's range""") as trk:
            self.play(self.frame.animate.rescale_to_fit(focus_obj.get_height() + 3, 1).move_to(focus_obj))
            split_weights([transformer6.embeddings], [transformer7.embeddings], RED, 1)

        # create allreduce
        l1 = transformer6.submobjects[1]
        l2 = transformer7.submobjects[1]
        all_reduce1 = FBlock("All Reduce\nSum", width=2.8, height=1.4, text_rotation_deg=0).move_to(Group(l1, l2))
        line_kwargs = dict(color=RED, width=0.3)
        c1 = Line3D(l1.get_center(), all_reduce1.get_top(), **line_kwargs)
        c2 = Line3D(all_reduce1.get_bottom(), l2.get_center(), **line_kwargs)
        with self.voiceover(text="""After running it, we perform an allreduce operation to synchronize our embeddings""") as trk:
            self.play(ShowCreation(c1), ShowCreation(c2), all_reduce1.create())
        
        t = transformer6.transformer_layers[0]
        t2 = transformer7.transformer_layers[0]

        with self.voiceover(text="""Next we have our RMS norm, for this one we just performe it on both GPU's, it doesn't have any weights
                            and is a memory bound operation so there is no speedups from spliting this""") as trk:
            self.play(self.frame.animate.shift((t2.rms_norm1.get_center()[0] - self.frame.get_center()[0]) * RIGHT))
            self.play(t2.rms_norm1.block.animate.set_color(YELLOW_E))

        # do mats
        t.set_mats()
        w = t.q_proj.w.copy()
        with self.voiceover(text="""Next we have our Q, K and V matrices, they are structured in a simillar way so let's just take Q as an example""") as trk:
            self.play(self.frame.animate.shift((t2.qkv_group.get_center()[0] - self.frame.get_center()[0]) * RIGHT))
            self.play(w.animate.rotate(radians(-25), DOWN).arrange(DOWN).move_to(Group(t.qkv_group, t2.qkv_group)))


        mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,h}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,h}"]])
        mat.move_to(w)
        with self.voiceover(text="""They have multiple weights for each head but the way that it runs in the background, is that all 
                            the head weights are stacked on top of each other and we run this as a single matrix multiplication""") as trk:
            self.play(ReplacementTransform(w, Group(mat)))

        #Create rowwise split 
        mat_up = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{\\frac{m}{2},0}", "w_{\\frac{m}{2},1}", "\\cdots", "w_{\\frac{m}{2},h}"]],
                           h_buff=1.15)

        mat_down = TexMatrix([["w_{\\frac{m}{2}+1,0}", "w_{\\frac{m}{2}+1,1}", "\\cdots", "w_{\\frac{m}{2}+1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,h}"]])
        Group(mat_up, mat_down).scale(0.8).arrange(DOWN).move_to(mat,aligned_edge=LEFT)

        mat_up.get_brackets()[0]
        with self.voiceover(text="""The way that we split this matrix is that we cut it in the middle, and give one part to one GPU and another to the second one""") as trk:
            self.play(ReplacementTransform(mat.get_brackets()[0], VGroup(mat_up.get_brackets()[0], mat_down.get_brackets()[0])),
                      ReplacementTransform(mat.get_brackets()[1], VGroup(mat_up.get_brackets()[1], mat_down.get_brackets()[1])),
                      ReplacementTransform(VGroup(*mat.elements[len(mat.elements)//2:]), VGroup(*mat_down.elements)),
                      ReplacementTransform(VGroup(*mat.elements[:len(mat.elements)//2]), VGroup(*mat_up.elements)), run_time=1)
            self.wait(1)
            self.play(FadeOut(mat_up), FadeOut(mat_down))

        with self.voiceover(text="""This is called a rowwise split, we split all 3 matrices this way""") as trk:
            split_weights([t.q_proj, t.k_proj, t.v_proj], [t2.q_proj, t2.k_proj, t2.v_proj], TEAL, dim=1)

        with self.voiceover(text="""RoPE as well as attention we also run independently across the GPUs, as the input to those will 
                            differ after our rowwise split so we don't repeat any calculations""") as trk:
            self.play(self.frame.animate.shift((t2.attn.get_center()[0] - self.frame.get_center()[0]) * RIGHT))
            self.play(t2.attn.block.animate.set_color(GOLD_E),
                      t2.rotary1.block.animate.set_color(BLUE),
                      t2.rotary2.block.animate.set_color(BLUE)
                      )

        #create input
        inp_up = TexMatrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,b}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{\\frac{m}{2},0}", "x_{\\frac{m}{2},1}", "\\cdots", "x_{\\frac{m}{2},b}"]],
                           h_buff=1.15)

        inp_down = TexMatrix([["x_{\\frac{m}{2}+1,0}", "x_{\\frac{m}{2}+1,1}", "\\cdots", "x_{\\frac{m}{2}+1,b}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{m,0}", "x_{m,1}", "\\cdots", "x_{m,b}"]])
        
        Group(inp_up, inp_down).arrange(DOWN,buff=1).move_to(Group(t.attn, t2.attn)).scale(0.7)
        with self.voiceover(text="""Next we have our out projection matrix""") as trk:
            self.play(self.frame.animate.shift((t2.out_proj.get_center()[0] - self.frame.get_center()[0]) * RIGHT))

        with self.voiceover(text="""And after our attention computation, each GPU has a different half of the input to this matrix""") as trk:
            self.play(ReplacementTransform(t.attn.t.copy(), inp_up), ReplacementTransform(t2.attn.t.copy(), inp_down))

        with self.voiceover(text="""We could do another allreduce but remember that this is a very slow operation, to the rescue comes an interesting observation""") as trk:
            pass

        #create colwise split
        mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,m}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,m}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{h,0}", "w_{h,1}", "\\cdots", "w_{h,m}"]])
        w = t.out_proj.w.copy()
        mat_left = TexMatrix([["w_{0,0}", "\\cdots", "w_{0,\\frac{m}{2}}"],
                                      ["w_{1,0}", "\\cdots", "w_{1,\\frac{m}{2}}"],
                                      ["\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{h,0}", "\\cdots", "w_{h,\\frac{m}{2}}"]], h_buff=1.15).scale(0.6)

        mat_right = TexMatrix([["w_{0,\\frac{m}{2}+1}", "\\cdots", "w_{0,n}"],
                              ["w_{1,\\frac{m}{2}+1}", "\\cdots", "w_{1,n}"],
                              ["\\vdots", "\\ddots", "\\vdots"],
                              ["w_{h,\\frac{m}{2}+1}", "\\cdots", "w_{h,n}"]]).scale(0.6)
        Group(mat_left, mat_right).arrange(RIGHT).next_to(Group(inp_up, inp_down), RIGHT)
        mat.move_to(Group(mat_left, mat_right))
        with self.voiceover(text="""We can take our input matrix from the out projection""") as trk:
            self.play(ReplacementTransform(w, mat))

        mat_up.get_brackets()[0]
        l = []
        r = []
        for i, e in enumerate(mat.elements):
            if i%4 < 2:
                l.append(e)
            else:
                r.append(e)
        with self.voiceover(text="""And split it across the column dimension""") as trk:
            self.play(ReplacementTransform(mat.get_brackets()[0], mat_left.get_brackets()[0]),
                      ReplacementTransform(mat.get_brackets()[1],  mat_right.get_brackets()[1]),
                      ShowCreation(mat_left.get_brackets()[1]),
                      ShowCreation(mat_right.get_brackets()[0]),
                      ReplacementTransform(VGroup(*l), VGroup(*mat_left.elements)),
                      ReplacementTransform(VGroup(*r), VGroup(*mat_right.elements)), run_time=2)

        #move matrices
        with self.voiceover(text="""Right now, the shapes of those matrices are compatible with our inputs""") as trk:
            self.play(Group(mat_left, mat_right).animate.arrange(DOWN).move_to(mat))

        #create output
        out_up = TexMatrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,h}"],
                                      ["x_{1,0}", "x_{1,1}", "\\cdots", "x_{1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{b,0}", "x_{b,1}", "\\cdots", "x_{b,h}"]]).scale(0.6).next_to(mat_left, RIGHT, buff=1)

        out_down = TexMatrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,h}"],
                                      ["x_{1,0}", "x_{1,1}", "\\cdots", "x_{1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{b,0}", "x_{b,1}", "\\cdots", "x_{b,h}"]]).scale(0.6).next_to(mat_right, RIGHT, buff=1)
        u = mat_left.copy()
        d = mat_right.copy()
        with self.voiceover(text="""And after we perform our matrix mutliplication""") as trk:
            self.play(ReplacementTransform(inp_up, u), ReplacementTransform(inp_down, d))
            self.play(ReplacementTransform(u, out_up), ReplacementTransform(d, out_down))

        x_sum = out_up.copy().move_to(Group(out_up, out_down))
        with self.voiceover(text="""We just need to sum both matrices for the final output""") as trk:
            self.play(ReplacementTransform(Group(out_up, out_down), Group(x_sum)))
            self.play(FadeOut(x_sum), FadeOut(mat_left), FadeOut(mat_right))

        #transfer data

        with self.voiceover(text="""That's why we split every second matrix columnwise""") as trk:
            split_weights([t.out_proj], [t2.out_proj], TEAL)
    
        # create allreduce
        def create_allreduce(t, t2, obj, obj2):
            l1 = t.submobjects[t.submobjects.index(obj) + 1]
            l2 = t2.submobjects[t2.submobjects.index(obj2) + 1]
            all_reduce2 = FBlock("All Reduce\nSum", width=2.8, height=1.4, text_rotation_deg=0).move_to(Group(l1, l2))
            line_kwargs = dict(color=RED, width=0.3)
            c3 = Line3D(l1.get_center(), all_reduce2.get_top(), **line_kwargs)
            c4 = Line3D(all_reduce2.get_bottom(), l2.get_center(), **line_kwargs)
            self.play(ShowCreation(c3), ShowCreation(c4), all_reduce2.create())
        with self.voiceover(text="""It allows us to do a GPU to GPU sync every second matrix multiplication""") as trk:
            create_allreduce(t, t2, t.out_proj, t2.out_proj)

        with self.voiceover(text="""And we do the same for the rest of the elements inside our transformer block""") as trk:
            self.play(self.frame.animate.shift((t2.swiglu.get_center()[0] - self.frame.get_center()[0]) * RIGHT))
            self.play(t2.residual1.block.animate.set_color(BLUE))
            self.play(t2.rms_norm2.block.animate.set_color(YELLOW_E))
            split_weights([t.ffn_gate, t.ffn_up], [t2.ffn_gate, t2.ffn_up], TEAL, dim=1)
            self.play(t2.swiglu.block.animate.set_color(BLUE))
            split_weights([t.ffn_down], [t2.ffn_down], TEAL)
            create_allreduce(t, t2, t.ffn_down, t2.ffn_down)
            self.play(t2.residual2.block.animate.set_color(BLUE))


        total_time = 36 # Take from print below
        total_distance = transformer7.get_right()[0] - self.frame.get_center()[0]
        start_time = self.time 
        def updater(m, dt):
            #TODO why is the updater called twice?
            dist = dt*total_distance/(2*total_time)
            self.frame.shift(dist*RIGHT)

        for i in range(1, len(transformer6.transformer_layers)):
            t = transformer6.transformer_layers[i]
            t2 = transformer7.transformer_layers[i]
            self.play(t2.rms_norm1.block.animate.set_color(YELLOW_E))
            split_weights([t.q_proj, t.k_proj, t.v_proj], [t2.q_proj, t2.k_proj, t2.v_proj], TEAL, dim=1)
            self.play(t2.attn.block.animate.set_color(GOLD_E),
                      t2.rotary1.block.animate.set_color(BLUE),
                      t2.rotary2.block.animate.set_color(BLUE))
            split_weights([t.out_proj], [t2.out_proj], TEAL)
            create_allreduce(t, t2, t.out_proj, t2.out_proj)
            self.play(t2.residual1.block.animate.set_color(BLUE))
            self.play(t2.rms_norm2.block.animate.set_color(YELLOW_E))
            split_weights([t.ffn_gate, t.ffn_up], [t2.ffn_gate, t2.ffn_up], TEAL, dim=1)
            self.play(t2.swiglu.block.animate.set_color(BLUE))
            split_weights([t.ffn_down], [t2.ffn_down], TEAL)
            create_allreduce(t, t2, t.ffn_down, t2.ffn_down)
            self.play(t2.residual2.block.animate.set_color(BLUE))

        print("creating took", self.time - start_time)
        self.frame.remove_updater(updater)


        with self.voiceover(text="""For the LM head we run RMS norm on both GPUs""") as trk:
            self.play(transformer7.rms_norm.animate.set_color(YELLOW_E))

        with self.voiceover(text="""We split the linear layer weights""") as trk:
            split_weights([transformer6.linear], [transformer7.linear], TEAL)

        with self.voiceover(text="""And we perform an allreduce operation to get the final output""") as trk:
            create_allreduce(transformer6, transformer7, transformer6.linear, transformer7.linear)

        with self.voiceover(text="""The output probabilites are calculated only on one GPU so we skip this layer on all other tensor parallel
                            ranks except for rank 0""") as trk:
            pass
