
import os
from manimlib import *
from math import radians
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene

class TensorCores(VoiceoverScene):
    def construct(self):
        # init scene
        self.set_speech_service(
            # RecorderService(transcription_model="base")
            GTTSService(transcription_model="base")
            )
        self.voiceovers_in_embed = True

        total_n = 32
        tile_n = 8
        self.play(*[FadeOut(x) for x in self.mobjects])
        mat1_f = [Square(stroke_width=0, fill_color=GREEN, fill_opacity=0.5) for _ in range(total_n*total_n)]
        mat2_f = [Square(stroke_width=0, fill_color=BLUE, fill_opacity=0.5) for _ in range(total_n*total_n)]
        mat3_f = [Square(stroke_width=0, fill_color=ORANGE, fill_opacity=0.5) for _ in range(total_n*total_n)]


        g1 = Group(*mat1_f).arrange_in_grid(total_n, total_n, buff=4).move_to(ORIGIN, aligned_edge=UL).shift(25*UP + 15*LEFT)
        g2 = Group(*mat2_f).arrange_in_grid(total_n, total_n, buff=4).next_to(g1, UP, buff = 2)
        g3 = Group(*mat3_f).arrange_in_grid(total_n, total_n, buff=4).next_to(g1, LEFT, buff = 2)

        mat1 = []
        mat2 = []
        mat3 = []
        for i in range(tile_n):
            for j in range(tile_n):
                r = i
                c = j
                mat1.append(mat1_f[r*total_n + c])
                r = total_n - tile_n + i
                c = j
                mat2.append(mat2_f[r*total_n + c])
                r = i
                c = total_n - tile_n + j
                mat3.append(mat3_f[r*total_n + c])

        #move to 3d
        mat1_3d_f = [VCube(fill_color=GREY, fill_opacity=0.1).move_to(x.get_center()) for x in mat1_f]
        mat2_3d_f = [VCube(fill_color=x.get_color(), fill_opacity=0.3).move_to(x.get_center()).shift(2*IN) for x in mat2_f]
        mat3_3d_f = [VCube(fill_color=x.get_color(), fill_opacity=0.3).move_to(x.get_center()).shift(2*IN) for x in mat3_f]

        #rotate matrices
        mat1_3d_f_g = VGroup(*mat1_3d_f)
        mat2_3d_f_g = VGroup(*mat2_3d_f)
        mat3_3d_f_g = VGroup(*mat3_3d_f)


        mat2_3d_f_g.rotate(radians(90), axis=LEFT, about_edge=DOWN)
        mat3_3d_f_g.rotate(radians(90), axis=DOWN, about_edge=RIGHT)

        #show index calculation
        mat2_3d_f_g.shift(4*IN).shift(4*UP)
        mat3_3d_f_g.shift(4*IN).shift(4*LEFT)

            
        n_tiles = total_n // tile_n

        mat1_tiles = []
        mat2_tiles = []
        mat3_tiles = []

        for tile_x in range(0, total_n, tile_n):
            outer_tile1 = []
            outer_tile2 = []
            outer_tile3 = []
            for tile_y in range(0, total_n, tile_n):
                tile1 = []
                tile2 = []
                tile3 = []
                for x in range(tile_n):
                    for y in range(tile_n):
                        r = tile_x + x
                        c = tile_y + y
                        tile1.append(mat1_3d_f[r*total_n + c])

                        r = total_n - tile_n - tile_x + x
                        c = tile_y + y
                        tile2.append(mat2_3d_f[r*total_n + c])

                        r = tile_x + x
                        c = total_n - tile_n - tile_y + y
                        tile3.append(mat3_3d_f[r*total_n + c])
                outer_tile1.append(tile1)
                outer_tile2.append(tile2)
                outer_tile3.append(tile3)
            mat1_tiles.append(outer_tile1)
            mat2_tiles.append(outer_tile2)
            mat3_tiles.append(outer_tile3)

        anims = []
        for x in range(n_tiles):
            for y in range(n_tiles):
                anims.append(VGroup(*mat1_tiles[x][y]).animate.shift(y*4*RIGHT + x*4*DOWN))
                anims.append(VGroup(*mat2_tiles[x][y]).animate.shift(y*4*RIGHT + x*4*IN))
                anims.append(VGroup(*mat3_tiles[x][y]).animate.shift(y*4*IN + x*4*DOWN))


        # Show tiling
        self.frame.set_shape(329, 187).move_to([-28.3, 8.62, -25.36]).set_euler_angles(-2.24045432,  1.17009916,  1.86961547)
        with self.voiceover(text="""In the previous episode we saw how given thre matrices divided into tiles""") as trk:
            self.play(FadeIn(mat1_3d_f_g), FadeIn(mat2_3d_f_g), FadeIn(mat3_3d_f_g))
            self.play(*anims)

        def to_green(step, total = 6):
            percentage_red = 1 - step/total
            red = int(255*percentage_red + 131 * (1-percentage_red)) 
            return f"#{red:02x}C167"

        #play full matmul
        with self.voiceover(text="""We can use tensor cores to perform a matrix multiplication on them. If you didn't watch that episode you 
                            might want to catch up on it, if you think you know enough about CUDA already and want to jump straight
                            to optimizations then stick around as in this episode we will implement hierarchical tiling to speed up our matmuls""") as trk:
            for tile_o in range(n_tiles):
                anims1 = []
                anims2 = [] 
                anims3 = []
                anims4 = [] 
                dot_prods = []
                cs = []
                for tile_i in range(n_tiles):
                    mat2_3d = mat2_tiles[tile_o][tile_i]
                    for tile_j in range(n_tiles):
                        mat3_3d = mat3_tiles[tile_j][tile_o]
                        mat1_3d = mat1_tiles[tile_j][tile_i]
                        anims1.extend([VGroup(*mat2_3d).animate.set_opacity(1), VGroup(*mat3_3d).animate.set_opacity(1)])
                        anims3.extend([VGroup(*mat2_3d).animate.set_opacity(0.3), VGroup(*mat3_3d).animate.set_opacity(0.3)])
                        dot_prod = []
                        for j in range(8):
                            for k in range(8):
                                v1 = [mat2_3d[i*8 + k] for i in range(8)]
                                v2 = [mat3_3d[j*8 + i] for i in range(8)]

                                for i in range(8):
                                    pos = mat1_3d[j*8 + k].get_center().copy()
                                    pos[2] = v1[i].get_center()[2] - v1[i].get_center()[2]
                                    dot_prod.append(VCube(fill_color=YELLOW, side_length=1).move_to(pos))

                        acc = VGroup(*dot_prod)

                        anims2.extend([Transform(VGroup(*mat2_3d).copy(), acc, remover=True),ReplacementTransform(VGroup(*mat3_3d).copy(), acc)])

                        #visualize accumulate
                        run_time = 0.5
                        mat_group = VGroup(*mat1_3d)
                        tmp = mat_group.copy().set_color(to_green(tile_o*8 + 7, total_n)).set_opacity(1)
                        anims4.extend([Transform(acc, tmp, remover=True), Transform(mat_group, tmp)])

                self.play(*anims1, run_time=run_time)
                self.play(*anims2, run_time=run_time)

                self.play(*anims4)
                self.play(*anims3)
        self.wait(1)
        self.play(*[FadeOut(x) for x in self.mobjects])
