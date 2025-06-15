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


class Parallelism(Scene):
    def construct(self):
        # self.set_speech_service(
        #     # RecorderService(transcription_model="base")
        #     GTTSService(transcription_model="base")
        #     )
        shader_dir = os.path.dirname(os.path.abspath(__file__)) + "/shaders/one_sided"
        Square3D.shader_folder = shader_dir

        class FBlock(Group):
            def __init__(self, text=None, formula=None, text_scale=1, weights=None, *args, **kwargs):
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
            def __init__(self, width, height):
                super().__init__()
                self.std_width = width 
                self.std_height = height 
                
                self.rms_norm1 = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width*2/3, height=self.std_height, color=YELLOW_E)
                
                self.rotary1 = FBlock("RoPE", width=self.std_width/3, height=self.std_height/3)
                self.rotary2 = FBlock("RoPE", width=self.std_width/3, height=self.std_height/3)

                self.q_proj = FBlock("Q Proj","Q = XW_q", width=self.std_width*2/3, height=self.std_height/3, color=TEAL)
                self.k_proj = FBlock("K Proj","K = XW_k", width=self.std_width*2/3, height=self.std_height/3, color=TEAL)
                self.v_proj = FBlock("V Proj","V = XW_v", width=self.std_width*2/3, height=self.std_height/3, color=TEAL)
                self.qkv_group = Group(
                        Group(self.q_proj, self.rotary1).arrange(RIGHT, buff=0.4).add(Connector(self.q_proj, self.rotary1, width=0.1, color=WHITE)) , 
                        Group(self.k_proj, self.rotary2).arrange(RIGHT, buff=0.4).add(Connector(self.k_proj, self.rotary2, width=0.1, color=WHITE)) , 
                        self.v_proj)
                self.qkv_group.arrange(DOWN, buff=0.5, aligned_edge=LEFT)

                self.attn = FBlock("Attention", "\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V",
                                   width=self.std_width, height=self.std_height, color=GOLD_E)

                self.up_proj = FBlock("Up Proj","Y = XW_v", width=self.std_width, height=self.std_height, color=TEAL)
                
                self.residual1 = FBlock("+", width=self.std_width//4, height=self.std_height//4)
                
                self.rms_norm2 = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width*2/3, height=self.std_height, color=YELLOW_E)
                
                self.ffn_gate = FBlock("Gate", "XW_g", width=self.std_width, height=self.std_height/2, color=TEAL)
                self.ffn_up = FBlock("Up Proj", "XW_u", width=self.std_width, height=self.std_height/2, color=TEAL)
                
                self.ffn_group = Group(self.ffn_gate, self.ffn_up)
                self.ffn_group.arrange(DOWN, buff=0.5)
                
                self.swiglu = FBlock("SwiGLU", r"x \cdot w \cdot \frac{1}{e^{-x}}",
                                     width=self.std_width/2, height=self.std_height)
                
                self.ffn_down =  FBlock("Down Proj", "XW_d", width=self.std_width, height=self.std_height, color=TEAL)
                self.residual2 = FBlock("+", width=self.std_width//4, height=self.std_height//4)
                
                self.add(self.rms_norm1)
                self.add(self.qkv_group)
                self.add(self.attn)
                self.add(self.up_proj)
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
                mat.move_to(self.up_proj).shift(2*RIGHT)
                self.up_proj.set_weights(mat)

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

                self.high_level = FBlock("Transformer\nBlock", width = self.get_width(), height = self.get_height(), text_scale=4)

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

            def duplicate_to(self, target):
                anims = []
                if self.is_hl:
                    anims.append(ReplacementTransform(self.high_level.copy(), target.high_level.move_to(target)))
                else:
                    for i, x in self.submobjects:
                        anims.append(ReplacementTransform(x.copy(), target.submobjects[i]))
                return AnimationGroup(*anims)

        class Transformer(Group):
            def __init__(self, std_width=4, std_height=4, num_blocks=4, *args, **kwargs):
                super().__init__()

                self.std_width = std_width
                self.std_height = std_height

                self.transformer_layers = []
                for _ in range(num_blocks):
                    self.transformer_layers.append(TransformerBlock(self.std_width, self.std_height))

                self.embeddings = FBlock("Embedding", width=self.std_width*1.5,
                                         height=self.transformer_layers[-1].get_height(), text_scale=2, color=RED)
                self.add(self.embeddings)

                for tb in self.transformer_layers:
                    self.add(tb)

                self.rms_norm = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width*2/3, height=self.std_height, color=YELLOW_E)
                self.linear = FBlock("Linear", "XW", width=self.std_width, height=self.std_height, color=TEAL)
                self.softmax = FBlock("Softmax", 
                                      r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}",
                                      width=self.std_width*2/3, height=self.std_height, color=GREEN)
                self.add(self.rms_norm)
                self.add(self.linear)
                self.add(self.softmax)

                self.arrange(RIGHT, buff=2)

                self.lines = []
                for x1, x2 in zip(self.submobjects, self.submobjects[1:]):
                    l = Connector(x1, x2, width=0.1, color=WHITE)
                    self.lines.append(l)

                for l, tb in zip(self.lines, self.transformer_layers):
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
                    anims.append(b.animate.shift(cum_dist*DOWN))

                cum_dist = 0
                for i, b in enumerate(self.lines):
                    if i % (len(self.lines)/n) == 0 and i != 0:
                        tmp = Dot().move_to(b.get_left() + cum_dist*DOWN)
                        cum_dist += dist
                        tmp2 = Dot().move_to(b.get_right() + cum_dist*DOWN)
                        #TODO this is too much hacks
                        anims.append(Transform(b, Connector(tmp, Group(tmp2), width=0.1, color=WHITE)))
                    else:
                        anims.append(b.animate.shift(cum_dist*DOWN))
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
                            m_c.set_rgba_array(rgba_s, name="stroke_rgba")
                
        # t.set_opacity(0)
        self.play(t.create(), self.frame.animate.match_width(t))

        # Create empty copy
        t2 = TransformerBlock(4,4).next_to(t, DOWN, buff=5)
        for smo in t2.get_family():
            if isinstance(smo, Prism):
                smo.set_color(GREY).set_opacity(1)
        self.play(t2.create())


        self.wait(1)



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

        t.set_mats()
        w = t.q_proj.w.copy()
        w.scale(0.5).move_to(Group(t.qkv_group, t2.qkv_group)).rotate(radians(-25), DOWN)
        self.add(w)
        w.arrange(DOWN).move_to(Group(t.qkv_group, t2.qkv_group))


        mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,h}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,h}"]])
        mat.move_to(w)
        self.play(ReplacementTransform(w, Group(mat)))

        #Create rowwise split 
        self.wait()

        mat_up = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{\\frac{m}{2},0}", "w_{\\frac{m}{2},1}", "\\cdots", "w_{\\frac{m}{2},h}"]],
                           h_buff=1.15)

        mat_down = TexMatrix([["w_{\\frac{m}{2}+1,0}", "w_{\\frac{m}{2}+1,1}", "\\cdots", "w_{\\frac{m}{2}+1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,h}"]])
        Group(mat_up, mat_down).scale(0.8).arrange(DOWN).move_to(mat,aligned_edge=LEFT)

        mat_up.get_brackets()[0]
        self.play(ReplacementTransform(mat.get_brackets()[0], VGroup(mat_up.get_brackets()[0], mat_down.get_brackets()[0])),
                  ReplacementTransform(mat.get_brackets()[1], VGroup(mat_up.get_brackets()[1], mat_down.get_brackets()[1])),
                  ReplacementTransform(VGroup(*mat.elements[len(mat.elements)//2:]), VGroup(*mat_down.elements)),
                  ReplacementTransform(VGroup(*mat.elements[:len(mat.elements)//2]), VGroup(*mat_up.elements)), run_time=5)
        self.wait(1)
        self.play(FadeOut(mat_up), FadeOut(mat_down))

        anims = []
        for b, b2 in zip([t.q_proj, t.k_proj, t.v_proj], [t2.q_proj, t2.k_proj, t2.v_proj]):

            down = b.copy().set_height(b.get_height()/2, True)
            down.move_to(b, aligned_edge=DOWN).set_color(TEAL).scale(1.01)

            up = b.copy().set_height(b.get_height()/2, True)
            up.move_to(b2, aligned_edge=DOWN).set_color(TEAL).scale(1.01)
            anims.append(Transform(down, up, remover=True))

            mid = b.get_center()[1]
            for smo in b.block.submobjects:
                points = smo.get_points()
                if len(points):
                    rgba = smo.data["rgba"].copy()
                    rgba[points[:, 1] < mid, :] = color_to_rgba(GREY)
                    smo.set_rgba_array(rgba)

        self.play(*anims)

        for b, b2 in zip([t.q_proj, t.k_proj, t.v_proj], [t2.q_proj, t2.k_proj, t2.v_proj]):
            mid = b2.get_center()[1]
            for smo in b2.block.submobjects:
                points = smo.get_points()
                if len(points):
                    rgba = smo.data["rgba"].copy()
                    rgba[points[:, 1] < mid, :] = color_to_rgba(TEAL)
                    smo.set_rgba_array(rgba)

        inp_up = TexMatrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,b}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{\\frac{m}{2},0}", "x_{\\frac{m}{2},1}", "\\cdots", "x_{\\frac{m}{2},b}"]],
                           h_buff=1.15)

        inp_down = TexMatrix([["x_{\\frac{m}{2}+1,0}", "x_{\\frac{m}{2}+1,1}", "\\cdots", "x_{\\frac{m}{2}+1,b}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{m,0}", "x_{m,1}", "\\cdots", "x_{m,b}"]])

        #create colwise split
        mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,m}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,m}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{h,0}", "w_{h,1}", "\\cdots", "w_{h,m}"]])
        w = t.up_proj.w.copy()
        mat.move_to(Group(t.up_proj, t2.up_proj))
        self.play(ReplacementTransform(w, mat))
        mat_left = TexMatrix([["w_{0,0}", "\\cdots", "w_{0,\\frac{m}{2}}"],
                                      ["w_{1,0}", "\\cdots", "w_{1,\\frac{m}{2}}"],
                                      ["\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{h,0}", "\\cdots", "w_{h,\\frac{m}{2}}"]], h_buff=1.15).scale(0.6)

        mat_right = TexMatrix([["w_{0,\\frac{m}{2}+1}", "\\cdots", "w_{0,n}"],
                              ["w_{1,\\frac{m}{2}+1}", "\\cdots", "w_{1,n}"],
                              ["\\vdots", "\\ddots", "\\vdots"],
                              ["w_{h,\\frac{m}{2}+1}", "\\cdots", "w_{h,n}"]]).scale(0.6)
        Group(mat_left, mat_right).arrange(RIGHT).move_to(mat)

        mat_up.get_brackets()[0]
        l = []
        r = []
        for i, e in enumerate(mat.elements):
            if i%4 < 2:
                l.append(e)
            else:
                r.append(e)
        self.play(ReplacementTransform(mat.get_brackets()[0], mat_left.get_brackets()[0]),
                  ReplacementTransform(mat.get_brackets()[1],  mat_right.get_brackets()[1]),
                  ShowCreation(mat_left.get_brackets()[1]),
                  ShowCreation(mat_right.get_brackets()[0]),
                  ReplacementTransform(VGroup(*l), VGroup(*mat_left.elements)),
                  ReplacementTransform(VGroup(*r), VGroup(*mat_right.elements)), run_time=5)


        #transfer data
        anims = []
        for b, b2 in zip([t.up_proj], [t2.up_proj]):

            down = b.copy().set_width(b.get_width()/2, True)
            down.move_to(b, aligned_edge=RIGHT).set_color(TEAL).scale(1.01)

            up = b.copy().set_width(b.get_width()/2, True)
            up.move_to(b2, aligned_edge=RIGHT).set_color(TEAL).scale(1.01)
            anims.append(Transform(down, up, remover=True))

            mid = b.get_center()[0]
            for smo in b.block.submobjects:
                points = smo.get_points()
                if len(points):
                    rgba = smo.data["rgba"].copy()
                    rgba[points[:, 0] > mid, :] = color_to_rgba(GREY)
                    smo.set_rgba_array(rgba)

        self.play(*anims)

        for b, b2 in zip([t.up_proj], [t2.up_proj]):
            mid = b2.get_center()[0]
            for smo in b2.block.submobjects:
                points = smo.get_points()
                if len(points):
                    rgba = smo.data["rgba"].copy()
                    rgba[points[:, 0] > mid, :] = color_to_rgba(TEAL)
                    smo.set_rgba_array(rgba)


