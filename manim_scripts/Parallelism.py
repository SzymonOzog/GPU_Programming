import os
from manimlib import *
from math import radians

from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene
import moderngl

CONNECTOR_WIDTH=0.25
hd_render = False
def trim_corner(mobject, corner, size):
    corn = mobject.get_corner(corner)

    dir1 = RIGHT if all(corner == UL) or all(corner == DL) else LEFT
    dir2 = DOWN if all(corner == UL) or all(corner == UR) else UP

    ptd = []
    for i, p in enumerate(mobject.get_points()):
        v = p - corn
        d0 = np.dot(v, dir1)
        d1 = np.dot(v, dir2)
        if d0 >= 0 and d1 >= 0 and (d0+d1) < size:
            ptd.append(i)
    mobject.data['rgba'][ptd, 3] = 0
    mobject.data['point'][ptd] = corn - (corner * np.array([size, size, size])) 
    for mob in mobject.family_members_with_points():
        mob.note_changed_data()

class MyBulletedList(VGroup):
    def __init__(
        self,
        *items: str,
        buff: float = MED_LARGE_BUFF,
        aligned_edge = LEFT,
        label = "",
        **kwargs
    ):
        labelled_content = [R"\item" + label + " " + item for item in items]
        tex_string = "\n".join([
            R"\begin{itemize}",
            *labelled_content,
            R"\end{itemize}"
        ])
        tex_text = TexText(tex_string, isolate=labelled_content, **kwargs)
        lines = (tex_text.select_part(part) for part in labelled_content)

        super().__init__(*lines)

        self.arrange(DOWN, buff=buff, aligned_edge=aligned_edge)

    def fade_all_but(self, index: int, opacity: float = 0.25, scale_factor=0.7) -> None:
        max_dot_height = max([item[0].get_height() for item in self.submobjects])
        for i, part in enumerate(self.submobjects):
            trg_dot_height = (1.0 if i == index else scale_factor) * max_dot_height
            part.set_fill(opacity=(1.0 if i == index else opacity))
            part.scale(trg_dot_height / part[0].get_height(), about_edge=LEFT)

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
    left[1] = min_y - CONNECTOR_WIDTH/2
    right = center.copy()
    right[1] = max_y + CONNECTOR_WIDTH/2
    return left, right

def connect(v_line, obj, up=True, *args, **kwargs):
    if is_grp(obj):
        lines = []
        for smo in obj.submobjects: 
            start = smo.get_left() if up else smo.get_right()
            end = v_line.get_center().copy()
            end[0] += -(CONNECTOR_WIDTH/2) if up else (CONNECTOR_WIDTH/2)
            end[1] = start[1]
            lines.append(Line3D(start, end, *args, **kwargs))
        return lines
    loc = obj.get_left() if up else obj.get_right()
    start = v_line.get_center()
    start[1] = loc[1] 
    return [Line3D(start, loc, *args, **kwargs)]

mats = []


class Parallelism(VoiceoverScene):
    samples=4
    def construct(self):
        self.set_speech_service(
            # RecorderService(transcription_model="base")
            GTTSService()
            )
        shader_dir = os.path.dirname(os.path.abspath(__file__)) + "/shaders/one_sided"
        shader_dir2  = os.path.dirname(os.path.abspath(__file__)) + "/shaders/one_sided2"
        Square3D.shader_folder = shader_dir
        Line3D.shader_folder = shader_dir2
        def updated_uniforms(cls):
            cls.uniforms: UniformDict = {
                "is_fixed_in_frame": 0.0,
                "shading": np.array(cls.shading, dtype=float),
                "clip_plane": np.zeros(4),
                "fade_shading": 0.0,
            }
        Square3D.init_uniforms = updated_uniforms
        hd_render = self.camera_config["resolution"][0] >= 1280

        class FBlock(Group):
            def __init__(self, text=None, formula=None, text_scale=1, text_rotation_deg=-90, weights=None, *args, **kwargs):
                super().__init__()
                self.block = Prism(square_resolution=(100, 100) if hd_render else (10,10),*args, **kwargs)
                self.t = None
                self.f = None
                self.add(self.block)
                self.showing_text = True
                if text is not None:
                    self.t = Text(text, font="Inter Light").move_to(self.block.get_corner(OUT)).rotate(radians(text_rotation_deg)).scale(text_scale)
                    t_h = self.t.get_height()
                    t_w = self.t.get_width()
                    b_h = self.block.get_height()
                    b_w = self.block.get_width()
                    if b_h/t_h > b_w/t_w:
                        self.t.match_width(self.block)
                        self.t.rescale_to_fit(b_w*0.8, dim=0)
                    else:
                        self.t.rescale_to_fit(b_h*0.8, dim=1)
                    self.add(self.t)
                if formula is not None:
                    self.f = Tex(formula).move_to(self.block.get_corner(OUT)).scale(text_scale)
                self.set_new_uniforms()

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
            
            def set_new_uniforms(self):
                for mob in self.get_family():
                    if len(mob.get_points()):
                        uniforms = {
                            "center": mob.get_center().astype(np.float32),
                            "shape": np.array(mob.get_shape()).astype(np.float32),
                            "fade_shading": 1.0,
                            }
                        mob.set_uniforms(uniforms)

            def note_changed_data(self):
                self.set_new_uniforms()
                super().note_changed_data()

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
                    resolution = (50,600) if hd_render else (21, 25)
                    self.v_line = Line3D(l, r, *args, resolution=resolution, **kwargs)
                    self.add(self.v_line)
                    self.bot = connect(self.v_line, start, False, *args, resolution=resolution, **kwargs)
                    self.top = connect(self.v_line, end, True, *args, resolution=resolution, **kwargs)
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

                resolution = (50,600) if hd_render else (21, 25)
                self.s_line = Line3D(start.get_top(), self.s_point+(CONNECTOR_WIDTH/2)*UP, resolution=resolution, *args, **kwargs)
                self.h_line = Line3D(self.s_point+(CONNECTOR_WIDTH/2)*LEFT, self.e_point+(CONNECTOR_WIDTH/2)*RIGHT, resolution=resolution, *args, **kwargs)
                self.e_line = Line3D(self.e_point+(CONNECTOR_WIDTH/2)*UP, end.get_top(), resolution=resolution, *args, **kwargs)

                trim_corner(self.s_line, UR, CONNECTOR_WIDTH*0.99)
                trim_corner(self.h_line, DL, CONNECTOR_WIDTH*0.99)
                trim_corner(self.h_line, DR, CONNECTOR_WIDTH*0.99)
                trim_corner(self.e_line, UL, CONNECTOR_WIDTH*0.99)

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
            def __init__(self, width, height, moe=False, *args, **kwargs):
                super().__init__()
                self.moe = moe
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
                        Group(self.q_proj, self.rotary1).arrange(RIGHT, buff=0.4).add(Connector(self.q_proj, self.rotary1, width=CONNECTOR_WIDTH, color=WHITE)) , 
                        Group(self.k_proj, self.rotary2).arrange(RIGHT, buff=0.4).add(Connector(self.k_proj, self.rotary2, width=CONNECTOR_WIDTH, color=WHITE)) , 
                        self.v_proj)
                self.qkv_group.arrange(DOWN, buff=0.5, aligned_edge=LEFT)

                self.attn = FBlock("Attention", "\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V",
                                   width=self.std_width, height=self.std_height, color=GOLD_E, *args, **kwargs)

                self.out_proj = FBlock("Out Proj","Y = XW_v", width=self.std_width, height=self.std_height, color=TEAL, *args, **kwargs)
                
                self.residual1 = FBlock("+", width=self.std_width//4, height=self.std_height//4)
                
                self.rms_norm2 = FBlock("RMS Norm", r"\frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}}",
                                        width=self.std_width*2/3, height=self.std_height, color=YELLOW_E)
                
                if moe:
                    self.moe_router = FBlock("Router", "XW_g", width=self.std_width*2/3, height=self.std_height, color=TEAL, *args, **kwargs)

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
                if moe:
                    self.add(self.moe_router)
                self.add(self.ffn_group)
                self.add(self.swiglu)
                self.add(self.ffn_down)
                self.add(self.residual2)

                self.arrange(RIGHT, buff=1.5)

                lines = []
                for x1, x2 in zip(self.submobjects, self.submobjects[1:]):
                    l = Connector(x1, x2, width=CONNECTOR_WIDTH, color=WHITE)
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
                                for _ in range(4)]).arrange(OUT, buff=0.5).rotate(radians(25), DOWN).scale(0.8) for _ in range(3)]
                Group(*mats).arrange(DOWN).move_to(self.k_proj).shift(8*RIGHT)
                self.q_proj.set_weights(mats[0])
                self.k_proj.set_weights(mats[1])
                self.v_proj.set_weights(mats[2])

                mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,m}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,m}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{h,0}", "w_{h,1}", "\\cdots", "w_{h,m}"]]).rotate(radians(25), DOWN).scale(1.5)
                mat.move_to(self.out_proj).shift(8*RIGHT)
                self.out_proj.set_weights(mat)

                mats = [TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,h}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{i,0}", "w_{i,1}", "\\cdots", "w_{i,h}"]]).rotate(radians(25), DOWN).scale(1.4) for _ in range(2)]
                Group(*mats).arrange(DOWN).move_to(self.ffn_group).shift(8*RIGHT)
                self.ffn_gate.set_weights(mats[0])
                self.ffn_up.set_weights(mats[1])

                mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,i}"],
                                      ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,i}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,i}"]]).rotate(radians(25), DOWN).scale(1.5)
                mat.move_to(self.ffn_down).shift(8*RIGHT)
                self.ffn_down.set_weights(mat)


            def create_residuals(self):
                res_y = self.rms_norm1.get_top()[1] + 1
                self.res = Residual(self.res_inp, self.residual1, res_y, width=CONNECTOR_WIDTH, color=WHITE)
                self.res2 = Residual(self.submobjects[self.submobjects.index(self.residual1) + 1], self.residual2, res_y, width=CONNECTOR_WIDTH, color=WHITE)

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
                    self.create_residuals()
                    self.deactivate_depth_test()
                    self.high_level.apply_depth_test()
                    ret = AnimationGroup(FadeOut(self.high_level), self.create(False), self.res.create(), self.res2.create())
                    self.is_hl = False
                else:
                    self.apply_depth_test()
                    self.high_level.deactivate_depth_test()
                    ret = AnimationGroup(FadeOut(self), self.create(True), FadeOut(self.res), FadeOut(self.res2))
                    self.is_hl = True
                return ret

            def duplicate_to(self, target, copy=True):
                anims = []
                if self.is_hl:
                    anims.append(ReplacementTransform(self.high_level.copy() if copy else self.high_level, target.high_level.move_to(target)))
                else:
                    for i, x in enumerate(self.submobjects):
                        anims.append(ReplacementTransform(x.copy() if copy else x, target.submobjects[i]))
                    if not hasattr(self, "res"):
                        self.create_residuals()
                    if not hasattr(target, "res"):
                        target.create_residuals()
                    anims.append(ReplacementTransform(self.res.copy() if copy else self.res, target.res))
                    anims.append(ReplacementTransform(self.res2.copy() if copy else self.res2, target.res2))
                return AnimationGroup(*anims)

            def trim_connectors(self):
                c1 = self.submobjects[self.submobjects.index(self.rms_norm1)+1]
                trim_corner(c1.v_line, UR, CONNECTOR_WIDTH*0.99)
                trim_corner(c1.v_line, DR, CONNECTOR_WIDTH*0.99)
                trim_corner(c1.top[0], DL, CONNECTOR_WIDTH*0.99)
                trim_corner(c1.top[-1], UL, CONNECTOR_WIDTH*0.99)

                c2 = self.submobjects[self.submobjects.index(self.moe_router if self. moe else self.rms_norm2)+1]
                trim_corner(c2.v_line, UR, CONNECTOR_WIDTH*0.99)
                trim_corner(c2.v_line, DR, CONNECTOR_WIDTH*0.99)
                trim_corner(c2.top[0], DL, CONNECTOR_WIDTH*0.99)
                trim_corner(c2.top[-1], UL, CONNECTOR_WIDTH*0.99)

        class Transformer(Group):
            def __init__(self, std_width=4, std_height=4, num_blocks=4, high_level=True, moe=False, *args, **kwargs):
                super().__init__()

                self.std_width = std_width
                self.std_height = std_height

                self.transformer_layers = []
                self.high_levels = []
                for _ in range(num_blocks):
                    self.transformer_layers.append(TransformerBlock(self.std_width, self.std_height, moe, *args, **kwargs))
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
                    l = Connector(x1, x2, width=CONNECTOR_WIDTH, color=WHITE)
                    self.lines.append(l)

                for l, tb, hl in zip(self.lines, self.transformer_layers, self.high_levels):
                    if high_level:
                        tb.move_to(hl)
                    # tb.create_residuals(l)
                    tb.res_inp = l
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
                self.trimmed = False
                # I have no idea why this messes things up if we do it on creation
                self.trim_connectors()

            def create(self, *args, **kwargs):
                anims = []
                for obj in self:
                    if hasattr(obj, "create"):
                        anims.append(obj.create(*args, **kwargs))
                    else:
                        anims.append(ShowCreation(obj, *args, **kwargs))
                return AnimationGroup(*anims)

            def trim_connectors(self):
                if self.trimmed:
                    return
                self.trimmed = True
                for l in self.transformer_layers:
                    l.trim_connectors()

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
                        transformed = Connector(tmp, Group(tmp2), width=CONNECTOR_WIDTH, color=WHITE)
                        anims.append(Uncreate(b))
                        anims.append(ShowCreation(transformed))
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
        global saved_colors, last_camera_x
        saved_colors = {}
        last_camera_x = float("inf")
        
        def updater(m, dt):
            global saved_colors, last_camera_x
            camera_x = self.frame.get_center()[0]
            if camera_x == last_camera_x:
                return
            last_camera_x = camera_x
            speed = 0.2
            residuals = [(t.res.get_family(True) + t.res2.get_family(True)) for t in transformer.transformer_layers]
            mobjects = (*transformer.get_family(True), *it.chain(*residuals))
            for mob in mobjects:
                points = mob.get_points()
                if len(points):
                    if "rgba" in mob.data_dtype.names:
                        if mob not in saved_colors:
                            saved_colors[mob] = mob.data["rgba"][..., :3].copy()
                        rgba = mob.data["rgba"].copy()
                        m_x_min = np.min(points[:, 0])
                        m_x_max = np.max(points[:, 0])
                        m_x_total = m_x_max - m_x_min
                        alpha = np.clip(((camera_x-points[:, 0])*speed), 0, 1)
                        rgba[:, 3] = -alpha

                        end = saved_colors[mob]
                        start = np.stack((color_to_rgb(GOLD),)*len(points))
                        new_color = start*(1 - alpha)[..., np.newaxis] + end*alpha[..., np.newaxis]

                        rgba[:, :3] = new_color
                        mob.set_rgba_array(rgba)
                    else:
                        rgba_s = mob.data["stroke_rgba"]#.copy()
                        rgba_f = mob.data["fill_rgba"]#.copy()
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
                MAT_START_OFFSET = (s.get_left()[0] - m_x_min)*2
                alpha = clip(camera_x - m_x_min, 0, 1)
                alpha2 = clip((camera_x - m_x_max)*speed, 0, 1)
                for m_s, m_c, m_e in zip(s.get_family(True), c.get_family(True), e.get_family(True)):
                    m_c = m_c.interpolate(m_s, m_e, alpha2)
                    if alpha <= 1:
                        points = m_c.get_points()
                        if len(points):
                            rgba_s = m_c.data["stroke_rgba"]#.copy()
                            rgba_f = m_c.data["fill_rgba"]#.copy()
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
                                    # saved_colors[mob] = color_to_rgb(mob.get_color())
                                    saved_colors[mob] = mob.data["rgba"][..., :3].copy()
                                rgba = mob.data["rgba"]#.copy()
                                m_x_min = np.min(points[:, 0])
                                m_x_max = np.max(points[:, 0])
                                m_x_total = m_x_max - m_x_min
                                start = np.stack((color_to_rgb(RED),)*len(points))
                                end = saved_colors[mob]
                                alpha = np.clip(np.abs(flash_x - points[:, 0])*0.1, 0.2, 1)
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
            for t in transformers:
                for mob in t.get_family(True):
                    points = mob.get_points()
                    if len(points):
                        if "rgba" in mob.data_dtype.names:
                            if mob not in saved_colors:
                                print("no color for", mob)
                            rgba = mob.data["rgba"]#.copy()
                            rgba[:, :3] = saved_colors[mob]
                            mob.set_rgba_array(rgba)
            self.frame.remove_updater(flash_updater)
                
        transformer = Transformer(4, 12, high_level=False).shift(5*DOWN)
        for t in transformer.transformer_layers:
            t.create_residuals()

        gpu0_f = SurroundingRectangle(transformer, buff=2, color=GREY)
        w = gpu0_f.get_width()
        h = gpu0_f.get_height()
        gpu0_bg = Square3D(color=GREY, resolution=(100, 100)).move_to(gpu0_f)
        gpu0_bg.rescale_to_fit(w, 0, stretch=True)
        gpu0_bg.rescale_to_fit(h, 1, stretch=True)
        center_x = gpu0_bg.get_center()[0]
        center_y = gpu0_bg.get_center()[1]

        rgba = gpu0_bg.data['rgba'].copy()
        points = gpu0_bg.get_points()
        rgba[:, 3] = 2*(np.maximum(np.abs(points[:, 0] - center_x)/w, np.abs(points[:, 1] - center_y)/h) ** 2)
        gpu0_bg.set_rgba_array(rgba, 'rgba')

        gpu0 = Group(gpu0_f, gpu0_bg)
        gpu0_t = Text("GPU0").set_color(GREEN).scale(10).rotate(radians(90)).next_to(gpu0, LEFT, buff=2)
        self.frame.rescale_to_fit(Group(gpu0, gpu0_t).get_width() + 10, dim=0)
        self.frame.save_state()

        for s,c,e,b in mats:
            Group(s,c,e).shift(5*DOWN)
            self.add(c)
        self.add(transformer, *[t.res for t in transformer.transformer_layers], *[t.res2 for t in transformer.transformer_layers])
        self.frame.set_euler_angles(-1.62773323,  0.46361119,  1.62378591).set_shape(1.8*28.885307, 1.8*16.235258).move_to([-102.19126   ,   -4.4578367 ,   0.18124883])        
        #animate creation
        transformer.set_opacity(0)
        self.frame.add_updater(updater)
        # self.play(self.frame.animate.shift(RIGHT * 20.05), run_time=1, rate_func=linear)

        # In the recent years there were a lot of advances in the LLM space, apart from just the architectural differences
        # the one big constant change that has been occuring, is increasing the model size. They went from milions to bilions
        # and now even trilions of parameters in size, taking up more and more space of our GPUs. This has sparked a lot of 
        # engineering efforts in making them run efficiently in a multi GPU setting. My name is Szymon and in this episode
        # I will present those methods and show you how to implement them. Let's get started
        self.play(self.frame.animate.shift(RIGHT * transformer.get_width() * 1.05), run_time=41, rate_func=linear)
        self.frame.remove_updater(updater)

        # Create GPU
        with self.voiceover(text="""In the simple case we have our model that is living on the GPU""") as trk:
            self.play(Restore(self.frame), transformer.transform([0,1,2,3]))
            self.play(ShowCreation(gpu0.submobjects[0]), ShowCreation(gpu0.submobjects[1]), Write(gpu0_t))

        cpu0_i = SVGMobject("./icons/cpu.svg").scale(8).set_color(WHITE).next_to(transformer, UP).shift(10*UP).set_color(BLUE)
        cpu0 = SurroundingRectangle(cpu0_i, buff=2, color=BLUE)
        cpu0_t = Text("CPU").set_color(BLUE).scale(10).next_to(cpu0, UP, aligned_edge=LEFT, buff=2)
        # Create CPU
        with self.voiceover(text="""And a CPU that is orchestrating it""") as trk:
            self.play(ShowCreation(cpu0_i), ShowCreation(cpu0), Write(cpu0_t))

        def trigger_gpu(gpu, on=True):
            return AnimationGroup(gpu.submobjects[0].animate.set_color(GREEN if on else GREY), 
                                  gpu.submobjects[1].animate.set_color(GREEN if on else GREY))
        class FadePartial(Fade):
            def create_target(self) -> Mobject:
                return self.mobject.copy()

            def create_starting_mobject(self) -> Mobject:
                start = super().create_starting_mobject()
                start.set_opacity(0.5)
                start.scale(1.0 / self.scale_factor)
                start.shift(-self.shift_vect)
                return start

        def x_y_path(
            start_points: np.ndarray,
            end_points: np.ndarray,
            alpha: float,
        ) -> np.ndarray:
            ret = start_points.copy()
            if alpha < 0.5:
                ret[:, 0] = interpolate(start_points[:, 0], end_points[:, 0], alpha * 2)
                return ret
            ret[:, 0] = end_points[:, 0]
            ret[:, 1] = interpolate(start_points[:, 1], end_points[:, 1], (alpha-0.5) * 2)
            return ret

        def y_x_path(
            start_points: np.ndarray,
            end_points: np.ndarray,
            alpha: float,
        ) -> np.ndarray:
            ret = start_points.copy()
            if alpha < 0.5:
                ret[:, 1] = interpolate(start_points[:, 1], end_points[:, 1], alpha * 2)
                return ret
            ret[:, 1] = end_points[:, 1]
            ret[:, 0] = interpolate(start_points[:, 0], end_points[:, 0], (alpha-0.5) * 2)
            return ret

        #run transformer
        with self.voiceover(text="""In a very simplified form the CPU is sending a task to process to the model,
                            getting the output, processing it and sending another task to our model living on the GPU""") as trk:
            while trk.get_remaining_duration() > 0:
                request = Square3D(color=RED, side_length=6).move_to(transformer.embeddings.get_left())
                self.play(FadeIn(request, shift=request.get_center() - cpu0_i.get_center(), path_func=x_y_path, remover=True),
                          trigger_gpu(gpu0, True),
                          run_time=2)
                run_transformers([transformer])
                request = Square3D(color=RED, side_length=6).move_to(cpu0_i)
                self.play(FadePartial(request, shift=request.get_left() - transformer.get_right(), path_func=y_x_path, remover=True) ,
                          trigger_gpu(gpu0, False),
                          run_time=2)


        # Create next GPU
        transformer2 = Transformer(4, 12).next_to(transformer, DOWN, buff=16)
        # gpu1 = SurroundingRectangle(transformer2, buff=2, color=GREY)
        gpu1 = gpu0.copy().move_to(transformer2)
        gpu1_t = Text("GPU1").set_color(GREEN).scale(10).rotate(radians(90)).next_to(gpu1, LEFT, buff=2)

        with self.voiceover(text="""But let's say we get another GPU and want to use this fact to speed up our inference""") as trk:
            self.play(ShowCreation(gpu1))
            self.play(Write(gpu1_t))


        # Data parallel
        with self.voiceover(text="""One simple thing to do would be to just make an entire copy of our model and put it on the second GPU""") as trk:
            self.play(transformer.duplicate_to(transformer2))

        # Name it
        dp_t = Text("Data Parallel").scale(50).next_to(cpu0, UP, buff=8).set_color(WHITE)
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
                    start_anims.extend([FadeIn(request_s, shift=request_s.get_center() - cpu0_i.get_center(), remover=True), trigger_gpu(gpu0, True), trigger_gpu(gpu1, True)])
                    request_e = Square3D(color=RED, side_length=6).move_to(cpu0_i) 
                    end_anims.extend([FadeIn(request_e, shift=request_e.get_center() - t.softmax.get_right(), remover=True), trigger_gpu(gpu0, False), trigger_gpu(gpu1, False)])

                self.play(*start_anims, run_time=2)
                run_transformers(transformers)
                self.play(*end_anims, run_time=2)

        gpus = [gpu0, gpu1]
        # Insufficient load
        with self.voiceover(text="""First of all, we might not have enough load, in this case the CPU is unable to 
                            schedule a task on both GPUs at once meaning that essentially one of those becomes idle""") as trk:
            while trk.get_remaining_duration() > 0:
                transformers = [transformer, transformer2]
                for t, g in zip(transformers, gpus):
                    request = Square3D(color=RED, side_length=6).move_to(t.embeddings.get_left())
                    self.play(FadeIn(request, shift=request.get_center() - cpu0_i.get_center(), path_func=x_y_path, remover=True),
                              trigger_gpu(g, True),
                              run_time=2)
                    run_transformers([t])
                    request = Square3D(color=RED, side_length=6).move_to(cpu0_i)
                    self.play(FadePartial(request, shift=request.get_left() - t.get_right() + RIGHT, path_func=y_x_path, remover=True) ,
                              trigger_gpu(g, False),
                              run_time=2)
        self.play(*[trigger_gpu(g, True) for g in gpus])

        
        with self.voiceover(text="""But the more popular reason for ditching data parallel is just the simple fact that the models of today
                            no longer fit on a single GPU""") as trk:
            pass

        # Make pp scene part
        cpu1 = cpu0.copy()
        cpu1_t = cpu0_t.copy()
        cpu1_i = cpu0_i.copy()
        transformer3 = Transformer(4, 12).move_to(transformer)
        gpu2 = gpu0.copy().set_color(GREY)
        gpu2_t = gpu0_t.copy()
        gpu3 = gpu1.copy().set_color(GREY)
        gpu3_t = gpu1_t.copy()
        cp = Group(cpu1, cpu1_t, cpu1_i, transformer3, gpu2, gpu2_t, gpu3, gpu3_t).next_to(
                Group(cpu0_t, gpu1), RIGHT, buff = 20)
        self.play(FadeIn(cp))


        with self.voiceover(text="""This has created a needs for methods that split our model parameters across multiple GPUs""") as trk:
            self.play(self.frame.animate.shift(gpu2.get_center() - gpu0.get_center()), run_time=trk.get_remaining_duration())

        # Pipeline parallel
        pp_t = Text("Pipeline Parallel").scale(50).next_to(cpu1, UP, buff=5).set_color(WHITE).align_to(dp_t, UP)
        dist = gpu2.get_center()[1] - gpu3.get_center() [1]
        with self.voiceover(text="""One such method would be Pipeline Parallelizm. In this setting, in this setting, a subset of 
                            our model layers is placed on one GPU  while the rest is placed on a different one""") as trk:
            self.play(Write(pp_t))
            self.play(transformer3.pipeline_parallelize(2, dist))


        with self.voiceover(text="""This effectively splits our model between two GPUs but introduces a few drawbacks""") as trk:
            while trk.get_remaining_duration() > 0:
                request = Square3D(color=RED, side_length=6).move_to(transformer3.embeddings.get_left())
                self.play(FadeIn(request, shift=request.get_center() - cpu1_i.get_center(), path_func=x_y_path, remover=True), run_time=2)
                a1 = AnimationGroup(trigger_gpu(gpu2, True), trigger_gpu(gpu3, False))
                a2 = AnimationGroup(trigger_gpu(gpu2, False), trigger_gpu(gpu3, True))
                a3 = AnimationGroup(trigger_gpu(gpu2, False), trigger_gpu(gpu3, False))
                run_transformers([transformer3], 1, [a1, a2, a3])
                request = Square3D(color=RED, side_length=6).move_to(cpu1_i)
                self.play(FadeIn(request, shift=request.get_center() - transformer3.softmax.get_right(), path_func=y_x_path, remover=True), run_time=2)

        with self.voiceover(text="""The most important of which is simillar to the issue we had in data parallel. 
                            When we don't have enough running requests, only one of our GPU's is doing the work, while the other one is idling""") as trk:
            while trk.get_remaining_duration() > 0:
                request = Square3D(color=RED, side_length=6).move_to(transformer3.embeddings.get_left())
                self.play(FadeIn(request, shift=request.get_center() - cpu1_i.get_center(), path_func=x_y_path, remover=True), run_time=2)
                a1 = AnimationGroup(trigger_gpu(gpu2, True), trigger_gpu(gpu3, False), run_time=2)
                a2 = AnimationGroup(trigger_gpu(gpu2, False), trigger_gpu(gpu3, True), run_time=2)
                a3 = AnimationGroup(trigger_gpu(gpu2, False), trigger_gpu(gpu3, False), run_time=2)
                run_transformers([transformer3], anim=[a1, a2, a3])
                request = Square3D(color=RED, side_length=6).move_to(cpu1_i)
                self.play(FadeIn(request, shift=request.get_center() - transformer3.softmax.get_right(), path_func=y_x_path, remover=True), run_time=2)

        self.play(*[trigger_gpu(g, True) for g in [gpu2, gpu3]])

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

        # start TP
        tp_t = Text("Tensor Parallel").scale(50).next_to(cpu2, UP, buff=5).set_color(WHITE).align_to(dp_t, UP)
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


        #duplicate transformers
        transformer6 = Transformer(4,4, high_level=False, text_rotation_deg=0).move_to(transformer4, aligned_edge=DOWN)
        transformer7 = Transformer(4,4, high_level=False, text_rotation_deg=0).move_to(transformer5, aligned_edge=UP)
        for mob in it.chain(transformer7.get_family(True), *[x.get_family(True) for x in transformer7.transformer_layers]):
            if isinstance(mob, Prism):
                mob.set_color(GREY)

        
        with self.voiceover(text="""And we split all model weights and calculations across all stages of the model""") as trk:
            pass

        def my_path_fn(
            start_points: np.ndarray,
            end_points: np.ndarray,
            alpha: float,
        ) -> np.ndarray:
            s_points = start_points.copy()
            e_points = end_points.copy()
            if alpha < 0.5:
                a = alpha*2
                w_alpha = 1 * (1 - a) + 0.01*a
            else:
                a = alpha*2 - 1
                w_alpha = 0.1 * (1 - a) + 1*a

            if len(start_points):
                interm = interpolate(s_points, e_points, alpha)
                center = (np.min(start_points[:, 1]) + np.max(end_points[:, 1]))/2
                total_dist = (np.min(start_points[:, 1]) - np.max(end_points[:, 1]))/2
                dist = np.abs(interm[:, 1] - center)
                dist_alpha = np.clip(dist/total_dist, a_min=0.1, a_max=1)
                final_alpha = rush_from(dist_alpha*w_alpha) 
                final_alpha=np.clip(final_alpha, a_min=0.2, a_max=1)

                min = np.min(start_points[:, 0])
                max = np.max(start_points[:, 0])
                center = (max+min)/2
                norm = start_points[:, 0] - center
                s_points[:, 0] = final_alpha*norm + center

                min = np.min(end_points[:, 0])
                max = np.max(end_points[:, 0])
                center = (max+min)/2
                norm = end_points[:, 0] - center
                e_points[:, 0] = norm*final_alpha + center

            return interpolate(s_points, e_points, alpha)

        def split_weights(t1_mobs, t2_mobs, color, dim=0, run_time=1):
            anims = []
            for b, b2 in zip(t1_mobs, t2_mobs):
                down = b.block.copy().rescale_to_fit(b.length_over_dim(dim)/2, dim, True)
                down.move_to(b, aligned_edge=DOWN if dim == 1 else RIGHT).set_color(color).scale(1.01)
                for mob in down.get_family():
                    mob.set_uniforms({
                            "center": mob.get_center().astype(np.float32),
                            "shape": np.array(mob.get_shape()).astype(np.float32),
                            "fade_shading": 1.0,
                            })

                up = b.block.copy().rescale_to_fit(b.length_over_dim(dim)/2, dim, True)
                up.move_to(b2, aligned_edge=DOWN if dim == 1 else RIGHT).set_color(color).scale(1.01)
                for mob in up.get_family():
                    mob.set_uniforms({
                            "center": mob.get_center().astype(np.float32),
                            "shape": np.array(mob.get_shape()).astype(np.float32),
                            "fade_shading": 1.0,
                            })
                anims.append(Transform(down, up, remover=True, path_func=my_path_fn))

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
        self.frame.save_state()

        # split embeddings
        with self.voiceover(text="""Going layer by layer, we first encounter our embeddings. The way we split them is that we just
                            divide them equally across all GPUs, each embedding only the tokens it has in it's range""") as trk:
            self.play(self.frame.animate.rescale_to_fit(focus_obj.get_height() + 3, 1).move_to(focus_obj),
                      transformer4.duplicate_to(transformer6, copy=False), transformer5.duplicate_to(transformer7, copy=False))
            split_weights([transformer6.embeddings], [transformer7.embeddings], RED, 1)

    
        # create allreduce
        def create_allreduce(t, t2, obj, obj2, run_time=1):
            l1 = t.submobjects[t.submobjects.index(obj) + 1]
            l2 = t2.submobjects[t2.submobjects.index(obj2) + 1]
            all_reduce2 = FBlock("All Reduce\nSum", width=2.8, height=1.4, text_rotation_deg=0).move_to(Group(l1, l2))
            line_kwargs = dict(color=RED, width=CONNECTOR_WIDTH)
            start = l1.get_center().copy()
            start[1] = gpu4.get_bottom()[1]
            c3 = Line3D(start, all_reduce2.get_top(), **line_kwargs)
            end = l2.get_center().copy()
            end[1] = gpu5.get_top()[1]
            c4 = Line3D(all_reduce2.get_bottom(), end, **line_kwargs)
            self.play(ShowCreation(c3), ShowCreation(c4), all_reduce2.create(), run_time=run_time)

        with self.voiceover(text="""After running it, we perform an allreduce operation to synchronize our embeddings""") as trk:
            create_allreduce(transformer6, transformer7, transformer6.embeddings, transformer7.embeddings)
        
        t = transformer6.transformer_layers[0]
        t2 = transformer7.transformer_layers[0]

        with self.voiceover(text="""Next we have our RMS norm, for this one we just performe it on both GPU's, it doesn't have any weights
                            and is a memory bound operation so there is no speedups from spliting this""") as trk:
            self.play(self.frame.animate.shift((t2.rms_norm1.get_center()[0] - self.frame.get_center()[0]) * RIGHT))
            self.play(t2.rms_norm1.block.animate.set_color(YELLOW_E))

        # do mats
        t.set_mats()
        w = t.q_proj.w.move_to(t.q_proj)
        with self.voiceover(text="""Next we have our Q, K and V matrices, they are structured in a simillar way so let's just take Q as an example""") as trk:
            self.play(self.frame.animate.shift((t2.qkv_group.get_center()[0] - self.frame.get_center()[0]) * RIGHT))
            self.play(w.animate.rotate(radians(-25), DOWN).arrange(DOWN).move_to(Group(t.qkv_group, t2.qkv_group)))


        # Transform to second matrix
        mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,h}"],
                         ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,h}"],
                         ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                         ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,h}"]]).scale(1.6)
        mat.move_to(w)
        with self.voiceover(text="""They have multiple weights for each head but the way that it runs in the background, is that all 
                            the head weights ar <bookmark mark='1'/>stacked on top of each other and we run this as a single matrix multiplication""") as trk:
            self.wait_until_bookmark("1")
            self.play(ReplacementTransform(VGroup(*[x.get_brackets()[0] for x in w.submobjects]), mat.get_brackets()[0]),
                      ReplacementTransform(VGroup(*[x.get_brackets()[1] for x in w.submobjects]), mat.get_brackets()[1]),
                      ReplacementTransform(VGroup(*w.submobjects[0].elements), VGroup(*mat.elements[:4])),
                      ReplacementTransform(VGroup(*w.submobjects[1].elements), VGroup(*mat.elements[4:8])),
                      ReplacementTransform(VGroup(*w.submobjects[2].elements), VGroup(*mat.elements[8:12])),
                      ReplacementTransform(VGroup(*w.submobjects[3].elements), VGroup(*mat.elements[12:])),
                      run_time=2
                      )

        #Create rowwise split 
        mat_up = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{\\frac{m}{2},0}", "w_{\\frac{m}{2},1}", "\\cdots", "w_{\\frac{m}{2},h}"]],
                           h_buff=1.15)

        mat_down = TexMatrix([["w_{\\frac{m}{2}+1,0}", "w_{\\frac{m}{2}+1,1}", "\\cdots", "w_{\\frac{m}{2}+1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,h}"]])
        Group(mat_up, mat_down).scale(1.6).arrange(DOWN).move_to(mat,aligned_edge=LEFT)

        mat_up.get_brackets()[0]
        with self.voiceover(text="""The way that we split this matrix is that we cut it in the middle, and give one part to one GPU and another to the second one""") as trk:
            self.play(ReplacementTransform(mat.get_brackets()[0], VGroup(mat_up.get_brackets()[0], mat_down.get_brackets()[0])),
                      ReplacementTransform(mat.get_brackets()[1], VGroup(mat_up.get_brackets()[1], mat_down.get_brackets()[1])),
                      ReplacementTransform(VGroup(*mat.elements[len(mat.elements)//2:]), VGroup(*mat_down.elements)),
                      ReplacementTransform(VGroup(*mat.elements[:len(mat.elements)//2]), VGroup(*mat_up.elements)), run_time=2, rate_func=rush_into)

        with self.voiceover(text="""This is called a rowwise split, we split all 3 matrices this way""") as trk:
            self.play(FadeOut(mat_up), FadeOut(mat_down))
            split_weights([t.q_proj, t.k_proj, t.v_proj], [t2.q_proj, t2.k_proj, t2.v_proj], TEAL, dim=1)

        with self.voiceover(text="""RoPE as well as attention we also run independently across the GPUs, as the input to those will 
                            differ after our rowwise split so we don't repeat any calculations""") as trk:
            self.play(self.frame.animate.shift((t2.attn.get_center()[0] - self.frame.get_center()[0]) * RIGHT))
            self.play(t2.attn.block.animate.set_color(GOLD_E),
                      t2.rotary1.block.animate.set_color(BLUE),
                      t2.rotary2.block.animate.set_color(BLUE),
                      run_time=2
                      )

        #create input
        inp_up = TexMatrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,b}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{\\frac{m}{2},0}", "x_{\\frac{m}{2},1}", "\\cdots", "x_{\\frac{m}{2},b}"]],
                           h_buff=1.15)

        inp_down = TexMatrix([["x_{\\frac{m}{2}+1,0}", "x_{\\frac{m}{2}+1,1}", "\\cdots", "x_{\\frac{m}{2}+1,b}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{m,0}", "x_{m,1}", "\\cdots", "x_{m,b}"]])
        
        Group(inp_up, inp_down).arrange(DOWN,buff=1).move_to(Group(t.attn, t2.attn)).scale(1.3)
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
                                      ["w_{h,0}", "w_{h,1}", "\\cdots", "w_{h,m}"]]).scale(1.6)
        w = t.out_proj.w.copy()
        mat_left = TexMatrix([["w_{0,0}", "\\cdots", "w_{0,\\frac{m}{2}}"],
                                      ["w_{1,0}", "\\cdots", "w_{1,\\frac{m}{2}}"],
                                      ["\\vdots", "\\ddots", "\\vdots"],
                                      ["w_{h,0}", "\\cdots", "w_{h,\\frac{m}{2}}"]], h_buff=1.15).scale(1.6)

        mat_right = TexMatrix([["w_{0,\\frac{m}{2}+1}", "\\cdots", "w_{0,n}"],
                              ["w_{1,\\frac{m}{2}+1}", "\\cdots", "w_{1,n}"],
                              ["\\vdots", "\\ddots", "\\vdots"],
                              ["w_{h,\\frac{m}{2}+1}", "\\cdots", "w_{h,n}"]]).scale(1.6)
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
            self.play(Group(mat_left, mat_right).animate.arrange(DOWN).next_to(Group(inp_up, inp_down), RIGHT))

        #create output
        out_up = TexMatrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,h}"],
                                      ["x_{1,0}", "x_{1,1}", "\\cdots", "x_{1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{b,0}", "x_{b,1}", "\\cdots", "x_{b,h}"]]).scale(1.6).next_to(mat_left, RIGHT, buff=1)

        out_down = TexMatrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,h}"],
                                      ["x_{1,0}", "x_{1,1}", "\\cdots", "x_{1,h}"],
                                      ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                      ["x_{b,0}", "x_{b,1}", "\\cdots", "x_{b,h}"]]).scale(1.6).next_to(mat_right, RIGHT, buff=1)
        u = mat_left.copy()
        d = mat_right.copy()
        with self.voiceover(text="""And after we perform our matrix mutliplication""") as trk:
            self.play(ReplacementTransform(inp_up, u), ReplacementTransform(inp_down, d))
            self.play(ReplacementTransform(u, out_up), ReplacementTransform(d, out_down))

        x_sum = out_up.copy().move_to(Group(out_up, out_down))
        with self.voiceover(text="""We just need to sum both matrices for the final output""") as trk:
            self.wait(3)
            self.play(ReplacementTransform(Group(out_up, out_down), Group(x_sum)))
            self.play(FadeOut(x_sum), FadeOut(mat_left), FadeOut(mat_right))

        #transfer data
        with self.voiceover(text="""That's why we split every second matrix columnwise""") as trk:
            split_weights([t.out_proj], [t2.out_proj], TEAL, run_time=2)

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


        # run to the end
        total_time = 15.3 # Take from print below
        total_distance = transformer7.get_right()[0] - self.frame.get_center()[0]
        start_time = self.time 
        def updater(m, dt):
            dist = dt*total_distance/total_time
            self.frame.shift(dist*RIGHT)
        self.frame.add_updater(updater)

        with self.voiceover(text="""the transformer blocks are stacked on top of each other, each models contaning usually several dozen of those
                            for each one in tensor parallel the split looks the same""") as trk:
            for i in range(1, len(transformer6.transformer_layers)):
                t = transformer6.transformer_layers[i]
                t2 = transformer7.transformer_layers[i]
                self.play(t2.rms_norm1.block.animate.set_color(YELLOW_E), run_time=0.5)
                split_weights([t.q_proj, t.k_proj, t.v_proj], [t2.q_proj, t2.k_proj, t2.v_proj], TEAL, dim=1, run_time=0.5)
                self.play(t2.attn.block.animate.set_color(GOLD_E),
                          t2.rotary1.block.animate.set_color(BLUE),
                          t2.rotary2.block.animate.set_color(BLUE), run_time=0.5)
                split_weights([t.out_proj], [t2.out_proj], TEAL, run_time=0.5)
                create_allreduce(t, t2, t.out_proj, t2.out_proj, run_time=0.3)
                self.play(t2.residual1.block.animate.set_color(BLUE), 
                          t2.rms_norm2.block.animate.set_color(YELLOW_E), run_time=0.5, lag_ratio=0.2)
                split_weights([t.ffn_gate, t.ffn_up], [t2.ffn_gate, t2.ffn_up], TEAL, dim=1, run_time=0.5)
                self.play(t2.swiglu.block.animate.set_color(BLUE), run_time=0.5)
                split_weights([t.ffn_down], [t2.ffn_down], TEAL, run_time=0.5)
                create_allreduce(t, t2, t.ffn_down, t2.ffn_down, run_time=0.3)
                self.play(t2.residual2.block.animate.set_color(BLUE), run_time=0.5)

        print("creating took", self.time - start_time)
        self.frame.remove_updater(updater)


        with self.voiceover(text="""For the LM head we run RMS norm on both GPUs""") as trk:
            self.play(transformer7.rms_norm.block.animate.set_color(YELLOW_E))

        with self.voiceover(text="""We split the linear layer weights""") as trk:
            split_weights([transformer6.linear], [transformer7.linear], TEAL)

        with self.voiceover(text="""And we perform an allreduce operation to get the final output""") as trk:
            create_allreduce(transformer6, transformer7, transformer6.linear, transformer7.linear)

        with self.voiceover(text="""The output probabilites are calculated only on one GPU so we skip this layer on all other tensor parallel
                            ranks except for rank 0""") as trk:
            pass

        transformer4 = Transformer(4, 12, high_level=False).move_to(gpu4)
        transformer5 = Transformer(4, 12, high_level=False).move_to(gpu5)
        for mob1, mob2 in zip(transformer6.get_family(), transformer4.get_family()):
            if len(mob1.get_points()) and "rgba" in mob1.data_dtype.names:
                mob2.set_rgba_array(mob1.data["rgba"].copy())

        for mob1, mob2 in zip(transformer7.get_family(), transformer5.get_family()):
            if len(mob1.get_points()) and "rgba" in mob1.data_dtype.names:
                mob2.set_rgba_array(mob1.data["rgba"].copy())

        with self.voiceover(text="""With this setup, we can run with a reduced memory footprint and with our GPUs working together
                            even with one request, this made the Tensor Parallel standard the most common standard across the industry""") as trk:
            self.play(Restore(self.frame),
                      transformer6.duplicate_to(transformer4, False),
                      transformer7.duplicate_to(transformer5, False))
            run_transformers(Group(transformer4, transformer5))

        
        # Make ep scene part
        cpu3 = cpu0.copy()
        cpu3_t = cpu0_t.copy()
        cpu3_i = cpu0_i.copy()
        gpu6 = gpu0.copy()
        gpu7 = gpu1.copy()
        transformer8 = Transformer(4, 12, high_level=False, moe=True).move_to(gpu6)
        transformer9 = Transformer(4, 12, high_level=False, moe=True).move_to(gpu7)
        gpu6.rescale_to_fit(transformer8.get_width()+2, dim=0)
        gpu7.rescale_to_fit(transformer8.get_width()+2, dim=0)
        gpu6_t = gpu0_t.copy().next_to(gpu6, LEFT, buff=2)
        gpu7_t = gpu1_t.copy().next_to(gpu7, LEFT, buff=2)
        cp3 = Group(cpu3, cpu3_t, cpu3_i, transformer8, transformer9, gpu6, gpu6_t, gpu7, gpu7_t).next_to(
                Group(cpu2_t, gpu5), RIGHT, buff = 20)


        for name, obj in transformer4.__dict__.items():
            if isinstance(obj, Mobject):
                for mob1, mob2 in zip(obj.get_family(), getattr(transformer8, name).get_family()):
                    if len(mob1.get_points()) and "rgba" in mob1.data_dtype.names:
                        mob2.set_rgba_array(mob1.data["rgba"].copy())

        for b, b2 in zip(transformer4.transformer_layers, transformer8.transformer_layers):
            b2.create_residuals()
            for name, obj in b.__dict__.items():
                if isinstance(obj, Mobject):
                    for mob1, mob2 in zip(obj.get_family(), getattr(b2, name).get_family()):
                        if len(mob1.get_points()) and "rgba" in mob1.data_dtype.names:
                            mob2.set_rgba_array(mob1.data["rgba"].copy())

        for name, obj in transformer5.__dict__.items():
            if isinstance(obj, Mobject):
                for mob1, mob2 in zip(obj.get_family(), getattr(transformer9, name).get_family()):
                    if len(mob1.get_points()) and "rgba" in mob1.data_dtype.names:
                        mob2.set_rgba_array(mob1.data["rgba"].copy())

        for b, b2 in zip(transformer5.transformer_layers, transformer9.transformer_layers):
            b2.create_residuals()
            for name, obj in b.__dict__.items():
                if isinstance(obj, Mobject):
                    for mob1, mob2 in zip(obj.get_family(), getattr(b2, name).get_family()):
                        if len(mob1.get_points()) and "rgba" in mob1.data_dtype.names:
                            mob2.set_rgba_array(mob1.data["rgba"].copy())

        self.wait(1)

        self.play(FadeIn(cp3))
        ep_t = Text("Expert Parallel").scale(50).next_to(cpu3, UP, buff=5).set_color(WHITE).align_to(dp_t, UP)
        with self.voiceover(text="""This type of model allowed us to parallelize by the expert dimension""") as trk:
            self.play(self.frame.animate.shift(gpu6.get_center() - gpu4.get_center()).rescale_to_fit(Group(gpu6, gpu6_t).get_width() + 10, dim=0) , 
                      run_time=trk.get_remaining_duration())
            self.play(Write(ep_t))

        # create allreduce 2
        def create_allreduce(t, t2, obj, obj2, run_time=1):
            l1 = t.submobjects[t.submobjects.index(obj) + 1]
            l2 = t2.submobjects[t2.submobjects.index(obj2) + 1]
            all_reduce2 = FBlock("All Reduce\nSum", width=2.8, height=1.4, text_rotation_deg=0).move_to(Group(l1, l2))
            line_kwargs = dict(color=RED, width=CONNECTOR_WIDTH)
            start = l1.get_center().copy()
            start[1] = gpu6.get_bottom()[1]
            c3 = Line3D(start, all_reduce2.get_top(), **line_kwargs)
            end = l2.get_center().copy()
            end[1] = gpu7.get_top()[1]
            c4 = Line3D(all_reduce2.get_bottom(), end, **line_kwargs)
            return AnimationGroup(ShowCreation(c3), ShowCreation(c4), all_reduce2.create(), run_time=run_time)

        anims = []
        for i in range(0, len(transformer6.transformer_layers)):
            t = transformer8.transformer_layers[i]
            t2 = transformer9.transformer_layers[i]

            anims.append(create_allreduce(t, t2, t.out_proj, t2.out_proj, run_time=0.3))
            anims.append(create_allreduce(t, t2, t.ffn_down, t2.ffn_down, run_time=0.3))

        with self.voiceover(text="""All of the communications required for tensor parallel are present""") as trk:
            self.play(LaggedStart(*anims))

        #create allreduce for routers
        anims = []
        for i in range(0, len(transformer6.transformer_layers)):
            t = transformer8.transformer_layers[i]
            t2 = transformer9.transformer_layers[i]
            anims.append(create_allreduce(t, t2, t.moe_router, t2.moe_router, run_time=0.3))
        with self.voiceover(text="""but we need to add another communication point after routing each experts""") as trk:
            self.play(LaggedStart(*anims))

        # Create bullet list
        line_start = (dp_t.get_center() + pp_t.get_center())/2 + 10*UP
        line_end = line_start + 800*DOWN
        table_line_v1 = Line(line_start, line_end, stroke_width=4)

        line_start = (tp_t.get_center() + pp_t.get_center())/2 + 10*UP
        line_end = line_start + 800*DOWN
        table_line_v2 = Line(line_start, line_end, stroke_width=4)

        line_start = (ep_t.get_center() + tp_t.get_center())/2 + 10*UP
        line_end = line_start + 800*DOWN
        table_line_v3 = Line(line_start, line_end, stroke_width=4)

        line_start = table_line_v1.get_top() - (table_line_v2.get_top() - table_line_v1.get_top())
        line_end = line_start + 800*DOWN
        table_line_v4 = Line(line_start, line_end, stroke_width=4)


        line_start = gpu1.get_bottom() + 20*DOWN + 310*LEFT
        H_LINE_LEN = 1130
        line_end = line_start + H_LINE_LEN*RIGHT
        table_line_h1 = Line(line_start, line_end, stroke_width=4)

        focus = Group(gpu0, gpu7, table_line_h1)
        with self.voiceover(text="""While Tensor Parallel is probably the most used across the industry, 
                            all different parallelism methods have their upsides and downsides""") as trk:
            self.play(self.frame.animate.rescale_to_fit(focus.get_width() + 20, dim=0).move_to(focus).align_to(pp_t, UP).shift(10*UP))

            self.play(ShowCreation(table_line_v1),
                      ShowCreation(table_line_v2),
                      ShowCreation(table_line_v3),
                      ShowCreation(table_line_v4),
                      ShowCreation(table_line_h1),
                      )

        # Compare implementation
        text_scale = 40
        text_buff = 15
        c1 = Text("Implementation").scale(text_scale).next_to(table_line_v4, LEFT, buff=text_buff).align_to(table_line_h1, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v4, table_line_v1).get_center()
        dp_c1 = Text("Easy").scale(text_scale).set_color(GREEN).move_to(loc).align_to(table_line_h1, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v2, table_line_v1).get_center()
        pp_c1 = Text("Medium").scale(text_scale).set_color(YELLOW).move_to(loc).align_to(table_line_h1, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v2, table_line_v1).get_center() + (table_line_v2.get_center() - table_line_v1.get_center())
        tp_c1 = Text("Hard").scale(text_scale).set_color(RED).move_to(loc).align_to(table_line_h1, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v3, table_line_v2).get_center() + (table_line_v3.get_center() - table_line_v2.get_center())
        ep_c1 = Text("Hard").scale(text_scale).set_color(RED).move_to(loc).align_to(table_line_h1, UP).shift(text_buff*DOWN)

        line_start = dp_c1.get_bottom() + text_buff*DOWN + 310*LEFT
        line_end = line_start + H_LINE_LEN*RIGHT
        table_line_h2 = Line(line_start, line_end, stroke_width=4)

        with self.voiceover(text="""Looking at implementation complexity""") as trk:
            self.play(Write(c1))
        with self.voiceover(text="""Data parallel is quite trivial, for a minimal working example
                            we can reuse single GPU code and just route our requests
                            to a different endpoint""") as trk:
            self.play(Write(dp_c1))

        with self.voiceover(text="""For pipeline parallel we would need to edit our codebase to initialize some distributed framework
                            and send data between GPUs""") as trk:
            self.play(Write(pp_c1))

        with self.voiceover(text="""Tensor and Expert Parallel are the most complicated to implement, it requires intergpu communication 
                            across many steps, and changes to how the forward pass of our model is implemented""") as trk:
            self.play(Write(tp_c1))
            self.play(Write(ep_c1))

        with self.voiceover(text="""With expert parallel there are even more challenges as efficient implementations, require us
                            to do load balancing of experts to ensure good GPU utilization""") as trk:
            pass

        # Compere memory reduction
        c2 = Text("Memory Reduction").scale(text_scale).next_to(table_line_v4, LEFT, buff=text_buff).align_to(table_line_h2, UP).shift(text_buff*DOWN)
        with self.voiceover(text="""Next we can compare how all methods reduce our memory footprint""") as trk:
            self.play(ShowCreation(table_line_h2))
            self.play(Write(c2))

        loc = Group(table_line_v4, table_line_v1).get_center()
        dp_c2 = Text("None").scale(text_scale).set_color(RED).move_to(loc).align_to(table_line_h2, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v2, table_line_v1).get_center()
        pp_c2 = Text("High").scale(text_scale).set_color(GREEN).move_to(loc).align_to(table_line_h2, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v2, table_line_v1).get_center() + (table_line_v2.get_center() - table_line_v1.get_center())
        tp_c2 = Text("High").scale(text_scale).set_color(GREEN).move_to(loc).align_to(table_line_h2, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v3, table_line_v2).get_center() + (table_line_v3.get_center() - table_line_v2.get_center())
        ep_c2 = Text("High").scale(text_scale).set_color(GREEN).move_to(loc).align_to(table_line_h2, UP).shift(text_buff*DOWN)
        with self.voiceover(text="""Data parallel doesn't reduce our memory requirements at all
                            <bookmark mark='1'/>while pipeline, tensor and expert parallel give us a big 
                            memory reduction
                            """) as trk:
            self.play(Write(dp_c2))
            self.wait_until_bookmark("1")
            self.play(Write(pp_c2))
            self.play(Write(tp_c2))
            self.play(Write(ep_c2))

        with self.voiceover(text="""With pipeline parallel there is one caveat, sometimmes it'self.
                            impossible to split weights across the GPUs equally so we end up with uneven memory 
                            consumption""") as trk:
            self.wait(3)
            self.play(pp_c2.animate.set_color(GREEN_E))

        # Compere speed 
        line_start = dp_c2.get_bottom() + text_buff*DOWN + 310*LEFT
        line_end = line_start + H_LINE_LEN*RIGHT
        table_line_h3 = Line(line_start, line_end, stroke_width=4)

        c3 = Text("Speedup").scale(text_scale).next_to(table_line_v4, LEFT, buff=text_buff).align_to(table_line_h3, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v4, table_line_v1).get_center()
        dp_c3 = Text("For big batches").scale(text_scale).set_color(RED).move_to(loc).align_to(table_line_h3, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v2, table_line_v1).get_center()
        pp_c3 = Text("For big batches").scale(text_scale).set_color(RED).move_to(loc).align_to(table_line_h3, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v2, table_line_v1).get_center() + (table_line_v2.get_center() - table_line_v1.get_center())
        tp_c3 = Text("For all batch sizes").scale(text_scale).set_color(GREEN).move_to(loc).align_to(table_line_h3, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v3, table_line_v2).get_center() + (table_line_v3.get_center() - table_line_v2.get_center())
        ep_c3 = Text("For medium batches").scale(text_scale).set_color(YELLOW).move_to(loc).align_to(table_line_h3, UP).shift(text_buff*DOWN)

        with self.voiceover(text="""Comparing speed benefits, we can see that <bookmark mark='1'/> data parallel and pipeline parallel
                            give us a speed benefit only for big batch sizes <bookmark mark='2'/> while tensor parallel increases the speed
                            since batch size of 1""") as trk:
            self.play(ShowCreation(table_line_h3))
            self.play(Write(c3))
            self.wait_until_bookmark("1")
            self.play(Write(dp_c3))
            self.play(Write(pp_c3))
            self.wait_until_bookmark("2")
            self.play(Write(tp_c3))

        with self.voiceover(text="""Expert parallel is a bit more tricky, we need to activate enough experts so that
                            our requests are routed to enough GPUs so the speedups usually hapen <bookmark mark='1'/>since medium sized batches""") as trk:
            self.wait_until_bookmark("1")
            self.play(Write(ep_c3))

        # Compere GPU communication 
        line_start = dp_c3.get_bottom() + text_buff*DOWN + 310*LEFT
        line_end = line_start + H_LINE_LEN*RIGHT
        table_line_h4 = Line(line_start, line_end, stroke_width=4)

        c4 = Text("GPU communication").scale(text_scale).next_to(table_line_v4, LEFT, buff=text_buff).align_to(table_line_h4, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v4, table_line_v1).get_center()
        dp_c4 = Text("None").scale(text_scale).set_color(GREEN).move_to(loc).align_to(table_line_h4, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v2, table_line_v1).get_center()
        pp_c4 = Text("Very little").scale(text_scale).set_color(YELLOW).move_to(loc).align_to(table_line_h4, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v2, table_line_v1).get_center() + (table_line_v2.get_center() - table_line_v1.get_center())
        tp_c4 = Text("High").scale(text_scale).set_color(RED).move_to(loc).align_to(table_line_h4, UP).shift(text_buff*DOWN)

        loc = Group(table_line_v3, table_line_v2).get_center() + (table_line_v3.get_center() - table_line_v2.get_center())
        ep_c4 = Text("Very high").scale(text_scale).set_color(RED).move_to(loc).align_to(table_line_h4, UP).shift(text_buff*DOWN)

        with self.voiceover(text="""Next we can compare communication needs, <bookmark mark='1'/> data parallel requires no GPU communication,""") as trk:
            self.play(ShowCreation(table_line_h4))
            self.play(Write(c4))
            self.wait_until_bookmark("1")
            self.play(Write(dp_c4))

        with self.voiceover(text="""pipeline parallel reqires us to send the activations only between 2 GPUs at the time and 
                            the number of synchronization points is equal to our pipeline parallel size""") as trk:
            self.play(Write(pp_c4))

        with self.voiceover(text="""Tensor and expoert parallel requires the most communication, as we have to synchronize our activations every second forward
                            layer and for routing in expert parallel""") as trk:
            self.play(Write(tp_c4))
            self.play(Write(ep_c4))

        # Compere maximum parallelism 
        line_start = dp_c4.get_bottom() + text_buff*DOWN + 310*LEFT
        line_end = line_start + H_LINE_LEN*RIGHT
        table_line_h5 = Line(line_start, line_end, stroke_width=4)
        c5 = Text("Parallelism limit").scale(text_scale).next_to(table_line_v4, LEFT, buff=text_buff).align_to(table_line_h5, UP).shift(text_buff*DOWN)
        loc = Group(table_line_v4, table_line_v1).get_center()
        dp_c5 = Text("None").scale(text_scale).set_color(GREEN).move_to(loc).align_to(table_line_h5, UP).shift(text_buff*DOWN)
        loc = Group(table_line_v2, table_line_v1).get_center()
        pp_c5 = Text("Compute blocks").scale(text_scale).set_color(YELLOW).move_to(loc).align_to(table_line_h5, UP).shift(text_buff*DOWN)
        loc = Group(table_line_v2, table_line_v1).get_center() + (table_line_v2.get_center() - table_line_v1.get_center())
        tp_c5 = Text("Matrix shape").scale(text_scale).set_color(RED).move_to(loc).align_to(table_line_h5, UP).shift(text_buff*DOWN)
        loc = Group(table_line_v3, table_line_v2).get_center() + (table_line_v3.get_center() - table_line_v2.get_center())
        ep_c5 = Text("No limit").scale(text_scale).set_color(GREEN).move_to(loc).align_to(table_line_h5, UP).shift(text_buff*DOWN)

        with self.voiceover(text="""As for the limitations on how much we can parallelize""") as trk:
            self.play(ShowCreation(table_line_h5))
            self.play(Write(c5))

        with self.voiceover(text="""Data parallel has no limitations, we can scale to infinity and beyond""") as trk:
            self.play(Write(dp_c5))

        with self.voiceover(text="""For pipeline parallel we are limited by the amount of compute blocks inside our model,
                            this is essentially a very artificial limit as it's not really something that we can hit before
                            we start getting bottlenecked by scheduling and communications""") as trk:
            self.play(Write(pp_c5))

        with self.voiceover(text="""For Tensor Parallel the limit is more tricky, it's the shape of the matrix that we are sharding across the GPUs
                            this can be very problematic as GPUs really like big matrix multiplications. When we shard them too much, we can start
                            seeing performance degradations because our GPUs are underutilized during matrix multiplication""") as trk:
            self.play(Write(tp_c5))

        with self.voiceover(text="""Expert parallelism solves this problem and lets us scale without any limits, we can always keep adding redundant experts""") as trk:
            self.play(Write(ep_c5))

        code = """import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl",
                        world_size=getenv("WORLD_SIZE"),
                        rank=getenv("RANK"),
                        )
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
data = torch.FloatTensor([1,] * 128).to("cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
value = data.mean().item()
world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"
"""
        # Switch to code scene
        code_obj = Code(code, alignment="LEFT")
        self.play(*[FadeOut(x) for x in self.mobjects])
        self.frame.rescale_to_fit(code_obj.get_width() + 10, dim=0).move_to(code_obj)


        with self.voiceover(text="""In terms of programming the communication, this is how the basic script
                            would look like in a high level framework like torch""") as trk:
            self.play(ShowCreation(code_obj))


        
        # create highlights
        import re
        code_s = code.replace("\n", "").replace(" ", "")
        s, e = re.search("backend=\"nccl\"", code_s).span()
        hl = SurroundingRectangle(Group(*code_obj.submobjects[s:e]), buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        with self.voiceover(text="""We have a few important components here, first would be the backend to use. This would be 
                            the library used for communication, in our case it's Nvidia's Collective Communications Library
                            so a library that works for GPU communication""") as trk:
            self.play(ShowCreation(hl))

        s, e = re.search(r'world_size=getenv\("WORLD_SIZE"\)', code_s).span()
        hl_t = SurroundingRectangle(Group(*code_obj.submobjects[s:e]), buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        with self.voiceover(text="""World size is how many processes, so in our case GPU's work together""") as trk:
            self.play(Transform(hl, hl_t))

        s, e = re.search(r'rank=getenv\("RANK"\)', code_s).span()
        hl_t = SurroundingRectangle(Group(*code_obj.submobjects[s:e]), buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        with self.voiceover(text="""Rank is the index of the GPU used by this process""") as trk:
            self.play(Transform(hl, hl_t))

        s, e = re.search(r'local_rank', code_s).span()
        hl_t = SurroundingRectangle(Group(*code_obj.submobjects[s:e]), buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        with self.voiceover(text="""and local rank would be localized to the current system, as there might be multiple 
                            nodes participating in our forward pass""") as trk:
            self.play(Transform(hl, hl_t))

        s, e = re.search(r'dist.all_reduce\(data,op=dist.ReduceOp.SUM\)', code_s).span()
        hl_t = SurroundingRectangle(Group(*code_obj.submobjects[s:e]), buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        with self.voiceover(text="""After initialization, we can now perform operations across GPU's like the forementioned allreduce""") as trk:
            self.play(Transform(hl, hl_t))
        
        self.wait(2)
        self.play(*[FadeOut(x) for x in self.mobjects])
