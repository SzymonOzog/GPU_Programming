import os
from manimlib import *
from math import radians

# from manim_voiceover.services.gtts import GTTSService
# from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from voicover_gl import VoiceoverScene
import moderngl

class Parallelism(Scene):
    def construct(self):
        # self.set_speech_service(
        #     # RecorderService(transcription_model="base")
        #     GTTSService(transcription_model="base")
        #     )
        shader_dir = os.path.dirname(os.path.abspath(__file__)) + "/shaders/one_sided"
        Square3D.shader_folder = shader_dir

        class FBlock(Group):
            def __init__(self, formula=None, *args, **kwargs):
                super().__init__()
                self.block = Prism(*args, **kwargs)
                
                self.add(self.block)
                if formula is not None:
                    self.t = Tex(formula).move_to(self.block.get_corner(OUT))
                    self.add(self.t)
                else:
                    self.t = None

            def create(self):
                anims = [ShowCreation(self.block)]
                if self.t:
                    anims.append(Write(self.t))
                return LaggedStart(*anims)

        class Connector(Group):
            def __init__(self, start, end, *args, **kwargs):
                super().__init__()
                self.l = Line(start, end, *args, **kwargs)
                self.add(self.l)
                self.args = args
                self.kwargs = kwargs
                # TODO why do we need this
                self.s = self.l.get_bottom()
                self.e = self.l.get_top()
            
            def create(self):
                return ShowCreation(self.l)

            def extend(self, dist):
                self.s += dist*DOWN
                self.e += dist*UP
                return Transform(self.l, Line(self.s, self.e, *self.args, **self.kwargs)) 

        class TransformerBlock(Group):
            def __init__(self):
                super().__init__()
                self.std_width = 8
                self.std_height = 1.5
                
                self.rms_norm1 = FBlock("\\text{RMSNorm}", width=self.std_width, height=self.std_height)
                
                self.q_proj = FBlock("Q = XW_q", width=self.std_width/3, height=self.std_height)
                self.k_proj = FBlock("K = XW_k", width=self.std_width/3, height=self.std_height)
                self.v_proj = FBlock("V = XW_v", width=self.std_width/3, height=self.std_height)
                self.qkv_group = Group(self.q_proj, self.k_proj, self.v_proj)
                self.qkv_group.arrange(RIGHT, buff=0.5)

                self.attn = FBlock("\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V", width=self.std_width, height=self.std_height)
                
                self.residual1 = FBlock("X + \\text{Attn}", width=self.std_width, height=self.std_height)
                
                self.rms_norm2 = FBlock("\\text{RMSNorm}", width=self.std_width, height=self.std_height)
                
                self.ffn_gate = FBlock("XW_g", width=self.std_width/2, height=self.std_height)
                self.ffn_up = FBlock("XW_u", width=self.std_width/2, height=self.std_height)
                
                self.ffn_group = Group(self.ffn_gate, self.ffn_up)
                self.ffn_group.arrange(RIGHT, buff=0.5)
                
                self.swiglu = FBlock("\\text{SwiGLU} = \\text{Swish}(XW_g) \\cdot XW_u", width=self.std_width, height=self.std_height)
                
                self.residual2 = FBlock("X + \\text{FFN}", width=self.std_width, height=self.std_height)
                
                self.rms_norm3 = FBlock("\\text{RMSNorm}", width=self.std_width, height=self.std_height)
                
                self.ffn_final = FBlock("XW_{out}", width=self.std_width, height=self.std_height)
                
                self.add(self.rms_norm1)
                self.add(self.qkv_group)
                self.add(self.attn)
                self.add(self.residual1)
                self.add(self.rms_norm2)
                self.add(self.ffn_group)
                self.add(self.swiglu)
                self.add(self.residual2)
                self.add(self.rms_norm3)
                self.add(self.ffn_final)

                self.arrange(UP, buff=1)

                lines = []
                for x1, x2 in zip(self.submobjects, self.submobjects[1:]):
                    l = Connector(x1, x2)
                    lines.append(l)
                i = len(self.submobjects) - 1
                for l in reversed(lines):
                    self.insert_submobject(i, l)
                    i -= 1

                self.high_level = FBlock("Transformer Block", width = self.get_width() * 1.05, height = self.get_height() * 1.05)

            def extend_at(self, obj, dist=2):
                idx = self.submobjects.index(obj)
                anims = []
                for i, smo in enumerate(self):
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
                    anims = []
                    for obj in self:
                        if hasattr(obj, "create"):
                            anims.append(obj.create())
                        else:
                            anims.append(ShowCreation(obj))
                    return LaggedStart(*anims)
                else:
                    return self.high_level.create()

            def transform(self):
                if self.is_hl:
                    ret = AnimationGroup(FadeOut(self), self.high_level.create())
                else:
                    ret = AnimationGroup(FadeOut(self.high_level), self.create())
                self.is_hl = not self.is_hl
                return ret

        t = TransformerBlock()
        self.play(t.create())
        self.play(t.transform())
        self.play(t.transform())
        self.play(t.extend_at(t.attn))
        self.play(t.shrink_at(t.attn))
