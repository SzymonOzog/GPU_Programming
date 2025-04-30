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

        class LlamaTransformerBlock(Group):
            def __init__(self):
                super().__init__()
                
                # Standard block width
                self.std_width = 8
                self.std_height = 1.5
                
                # Create all components
                self.rms_norm1 = FBlock("\\text{RMSNorm}", width=self.std_width, height=self.std_height)
                
                # Q, K, V projections (side by side)
                self.q_proj = FBlock("Q = XW_q", width=self.std_width/3, height=self.std_height)
                self.k_proj = FBlock("K = XW_k", width=self.std_width/3, height=self.std_height)
                self.v_proj = FBlock("V = XW_v", width=self.std_width/3, height=self.std_height)
                
                # Group QKV projections
                self.qkv_group = Group(self.q_proj, self.k_proj, self.v_proj)
                self.qkv_group.arrange(RIGHT, buff=0.5)

                # Attention block
                self.attn = FBlock("\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V", width=self.std_width, height=self.std_height)
                
                # First residual addition
                self.residual1 = FBlock("X + \\text{Attn}", width=self.std_width, height=self.std_height)
                
                # Second RMS norm
                self.rms_norm2 = FBlock("\\text{RMSNorm}", width=self.std_width, height=self.std_height)
                
                # FFN - SwiGLU
                self.ffn_gate = FBlock("XW_g", width=self.std_width/2, height=self.std_height)
                self.ffn_up = FBlock("XW_u", width=self.std_width/2, height=self.std_height)
                
                # Group FFN gate and up
                self.ffn_group = Group(self.ffn_gate, self.ffn_up)
                self.ffn_group.arrange(RIGHT, buff=0.5)
                
                # SwiGLU activation
                self.swiglu = FBlock("\\text{SwiGLU} = \\text{Swish}(XW_g) \\cdot XW_u", width=self.std_width, height=self.std_height)
                
                # Second residual addition
                self.residual2 = FBlock("X + \\text{FFN}", width=self.std_width, height=self.std_height)
                
                # Final RMS norm
                self.rms_norm3 = FBlock("\\text{RMSNorm}", width=self.std_width, height=self.std_height)
                
                # Final FFN
                self.ffn_final = FBlock("XW_{out}", width=self.std_width, height=self.std_height)
                
                # Add all blocks to the main group
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
                
                # Arrange blocks vertically
                self.arrange(DOWN, buff=1.2)
                
                # Create connections between blocks
                self.lines = []
                
                # Helper function to connect blocks
                def connect_blocks(block1, block2):
                    if isinstance(block1, Group) and not isinstance(block1, FBlock):
                        start = block1.get_bottom()
                    else:
                        start = block1.get_bottom()
                        
                    if isinstance(block2, Group) and not isinstance(block2, FBlock):
                        end = block2.get_top()
                    else:
                        end = block2.get_top()
                        
                    line = Line(start, end)
                    self.lines.append(line)
                    return line
                    
                # Connect RMS Norm1 to QKV group with fork
                main_line = Line(self.rms_norm1.get_bottom(), 
                                 self.rms_norm1.get_bottom() + DOWN * 0.4)
                self.lines.append(main_line)
                
                # Create forked lines to QKV projections
                for proj in [self.q_proj, self.k_proj, self.v_proj]:
                    fork = Line(main_line.get_end(), 
                                proj.get_top())
                    self.lines.append(fork)
                
                # Connect QKV to Attention with fork
                qkv_bottom = DOWN * 0.4 + self.qkv_group.get_bottom()
                line_to_qkv = Line(self.q_proj.get_bottom(), qkv_bottom)
                self.lines.append(line_to_qkv)
                line_to_qkv2 = Line(self.k_proj.get_bottom(), qkv_bottom)
                self.lines.append(line_to_qkv2)
                line_to_qkv3 = Line(self.v_proj.get_bottom(), qkv_bottom)
                self.lines.append(line_to_qkv3)
                
                line_qkv_attn = Line(qkv_bottom, self.attn.get_top())
                self.lines.append(line_qkv_attn)
                
                # Connect rest of blocks linearly
                self.lines.append(connect_blocks(self.attn, self.residual1))
                self.lines.append(connect_blocks(self.residual1, self.rms_norm2))
                
                # Connect RMS Norm2 to FFN group with fork
                ffn_line = Line(self.rms_norm2.get_bottom(), 
                               self.rms_norm2.get_bottom() + DOWN * 0.4)
                self.lines.append(ffn_line)
                
                # Create forked lines to FFN projections
                for proj in [self.ffn_gate, self.ffn_up]:
                    fork = Line(ffn_line.get_end(), 
                                proj.get_top())
                    self.lines.append(fork)
                
                # Connect FFN to SwiGLU with fork
                ffn_bottom = DOWN * 0.4 + self.ffn_group.get_bottom()
                line_to_ffn1 = Line(self.ffn_gate.get_bottom(), ffn_bottom)
                self.lines.append(line_to_ffn1)
                line_to_ffn2 = Line(self.ffn_up.get_bottom(), ffn_bottom)
                self.lines.append(line_to_ffn2)
                
                line_ffn_swiglu = Line(ffn_bottom, self.swiglu.get_top())
                self.lines.append(line_ffn_swiglu)
                
                # Connect remaining blocks
                self.lines.append(connect_blocks(self.swiglu, self.residual2))
                self.lines.append(connect_blocks(self.residual2, self.rms_norm3))
                self.lines.append(connect_blocks(self.rms_norm3, self.ffn_final))
                
                # Add the lines to the group
                for line in self.lines:
                    self.add(line)
                    
                # Also create a high-level representation
                self.high_level = FBlock("Llama Transformer Block", width=self.get_width() * 1.1, 
                                         height=self.get_height() * 1.1)
                self.is_hl = False
                
            def extend_at(self, obj, dist=2):
                """Extend the distance at a specific object"""
                idx = self.submobjects.index(obj)
                anims = []
                
                # Handle if obj is a group
                if isinstance(obj, Group) and not isinstance(obj, FBlock):
                    # Find the first submobject in the group
                    for i, smo in enumerate(self.submobjects):
                        if smo == obj:
                            idx = i
                            break
                            
                # Get indices of lines before and after the object
                line_before = None
                line_after = None
                
                for i, smo in enumerate(self.submobjects):
                    if isinstance(smo, Line):
                        if i < idx:
                            line_before = i
                        elif i > idx and line_after is None:
                            line_after = i
                            break
                
                # Create animations
                for i, smo in enumerate(self.submobjects):
                    if i <= idx:
                        anims.append(smo.animate.shift(dist*DOWN))
                    elif i == line_after:
                        # Transform the line after the object
                        old_line = smo
                        start_point = old_line.get_start() + dist*DOWN
                        end_point = old_line.get_end()
                        new_line = Line(start_point, end_point)
                        anims.append(Transform(old_line, new_line))
                    else:
                        anims.append(smo.animate.shift(0))  # Don't move other objects
                        
                return AnimationGroup(*anims)
                
            def create(self, high_level=True):
                """Create animation for the transformer block"""
                self.is_hl = high_level
                if high_level:
                    anims = []
                    for obj in self.submobjects:
                        if isinstance(obj, FBlock):
                            anims.append(obj.create())
                        elif isinstance(obj, Group):
                            # Handle nested groups like qkv_group
                            group_anims = []
                            for sub_obj in obj:
                                if isinstance(sub_obj, FBlock):
                                    group_anims.append(sub_obj.create())
                                else:
                                    group_anims.append(ShowCreation(sub_obj))
                            anims.append(LaggedStart(*group_anims))
                        else:
                            anims.append(ShowCreation(obj))
                    return LaggedStart(*anims)
                else:
                    return self.high_level.create()
                    
            def transform(self):
                """Transform between high-level and detailed view"""
                if self.is_hl:
                    ret = AnimationGroup(FadeOut(self), self.high_level.create())
                else:
                    ret = AnimationGroup(FadeOut(self.high_level), self.create())
                self.is_hl = not self.is_hl
                return ret
                
            def add_skip_connection(self, from_block, to_block, offset=0.5):
                """Add a skip/residual connection between blocks"""
                if isinstance(from_block, Group) and not isinstance(from_block, FBlock):
                    start = from_block.get_right() + RIGHT * offset
                else:
                    start = from_block.get_right() + RIGHT * offset
                    
                if isinstance(to_block, Group) and not isinstance(to_block, FBlock):
                    end = to_block.get_right() + RIGHT * offset
                else:
                    end = to_block.get_right() + RIGHT * offset
                    
                # Create a curved path for the skip connection
                points = [start]
                points.append(start + RIGHT * offset)
                midpoint = (start + end) / 2 + RIGHT * offset
                points.append(midpoint)
                points.append(end + RIGHT * offset)
                points.append(end)
                
                curve = VMobject()
                curve.set_points_smoothly(points)
                
                self.add(curve)
                return curve

        # t = TransformerBlock()
        t = LlamaTransformerBlock()
        self.play(t.create())
        self.play(t.transform())
        self.play(t.transform())
        # self.play(t.extend_at(t.attn))
