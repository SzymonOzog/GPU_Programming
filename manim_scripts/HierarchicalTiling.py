
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

        total_n = 64*2
        visible_n = 32
        tile_n = 8
        n_tiles = total_n // tile_n
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
        anims2 = []
        visible_tiles = visible_n//tile_n
        for x in range(n_tiles):
            for y in range(n_tiles):
                if x < visible_tiles and y < visible_tiles:
                    anims.append(FadeIn(VGroup(*mat1_tiles[x][y])))
                    anims.append(FadeIn(VGroup(*mat2_tiles[x][y])))
                    anims.append(FadeIn(VGroup(*mat3_tiles[x][y])))
                    anims2.append(VGroup(*mat1_tiles[x][y]).animate.shift(y*4*RIGHT + x*4*DOWN))
                    anims2.append(VGroup(*mat2_tiles[x][y]).animate.shift(y*4*RIGHT + x*4*IN))
                    anims2.append(VGroup(*mat3_tiles[x][y]).animate.shift(y*4*IN + x*4*DOWN))
                else:
                    VGroup(*mat1_tiles[x][y]).shift(y*4*RIGHT + x*4*DOWN)
                    VGroup(*mat2_tiles[x][y]).shift(y*4*RIGHT + x*4*IN)
                    VGroup(*mat3_tiles[x][y]).shift(y*4*IN + x*4*DOWN)


        # Show tiling
        self.frame.set_shape(329, 187).move_to([-28.3, 8.62, -25.36]).set_euler_angles(-2.24045432,  1.17009916,  1.86961547)
        with self.voiceover(text="""In the previous episode we saw how given thre matrices divided into tiles""") as trk:
            self.play(*anims)
            self.play(*anims2)

        def to_green(step, total = 6):
            percentage_red = 1 - step/total
            red = int(255*percentage_red + 131 * (1-percentage_red)) 
            return f"#{red:02x}C167"

        #play full matmul
        with self.voiceover(text="""We can use tensor cores to perform a matrix multiplication on them. If you didn't watch that episode you 
                            might want to catch up on it, if you think you know enough about CUDA already and want to jump straight
                            to optimizations then stick around as in this episode we will implement hierarchical tiling to speed up our matmuls""") as trk:
            for tile_o in range(visible_tiles):
                anims1 = []
                anims2 = [] 
                anims3 = []
                anims4 = [] 
                dot_prods = []
                cs = []
                for tile_i in range(visible_tiles):
                    mat2_3d = mat2_tiles[tile_o][tile_i]
                    for tile_j in range(visible_tiles):
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
                        tmp = mat_group.copy().set_color(to_green(tile_o*8 + 7, visible_n)).set_opacity(1)
                        anims4.extend([Transform(acc, tmp, remover=True), Transform(mat_group, tmp)])

                self.play(*anims1, run_time=run_time)
                self.play(*anims2, run_time=run_time)

                self.play(*anims4)
                self.play(*anims3)

        tiles1 = []
        tiles2 = []
        tiles3 = []

        w = VGroup(*mat1_tiles[0][0]).get_width()
        h = VGroup(*mat1_tiles[0][0]).get_height()
        d = VGroup(*mat1_tiles[0][0]).get_depth()

        for x in range(n_tiles):
            row1 = []
            row2 = []
            row3 = []
            for y in range(n_tiles):
                tile1_center = VGroup(*mat1_tiles[x][y]).get_center()
                tile2_center = VGroup(*mat2_tiles[x][y]).get_center()
                tile3_center = VGroup(*mat3_tiles[x][y]).get_center()
                
                cube1 = VPrism(width=w, height=h, depth=d, fill_color=GREEN, fill_opacity=0.3).move_to(tile1_center)
                cube2 = VPrism(width=w, height=d, depth=h, fill_color=BLUE, fill_opacity=0.3).move_to(tile2_center)
                cube3 = VPrism(width=d, height=h, depth=h, fill_color=ORANGE, fill_opacity=0.3).move_to(tile3_center)
                
                row1.append(cube1)
                row2.append(cube2)
                row3.append(cube3)
            
            tiles1.append(row1)
            tiles2.append(row2)
            tiles3.append(row3)

        transforms = []
        anims=[]

        for x in range(n_tiles):
            for y in range(n_tiles):
                if x < visible_tiles and y < visible_tiles:
                    transforms.append(ReplacementTransform(VGroup(*mat1_tiles[x][y]), tiles1[x][y]))
                    transforms.append(ReplacementTransform(VGroup(*mat2_tiles[x][y]), tiles2[x][y]))
                    transforms.append(ReplacementTransform(VGroup(*mat3_tiles[x][y]), tiles3[x][y]))
                else:
                    anims.append(FadeIn(tiles1[x][y]))
                    anims.append(FadeIn(tiles2[x][y]))
                    anims.append(FadeIn(tiles3[x][y]))


        
        # Play transformation animation
        with self.voiceover(text="""Since we are using tensor cores, we can abstract away single elements and start thinking of our 
                            matrices in terms of tiles""") as trk:
            self.play(*transforms, run_time=2)

        # Now scale up to more tiles
        with self.voiceover(text="""Let's start by thinking of an even bigger matrix""") as trk:
            self.play(*anims, self.frame.animate.set_shape(1160.2727, 659.4865))



        # show block tiles 
        anims = []
        for x in range(0, n_tiles, 2):
            for y in range(0, n_tiles, 2):
                dist_x = x*4 + 12*(x//4) + 30 *(x//8)
                dist_y = y*4 + 12*(y//4) + 30 *(x//8)
                anims.append(VGroup(*tiles1[x][y]).animate.shift(dist_y*RIGHT + dist_x*DOWN))
                anims.append(VGroup(*tiles2[x][y]).animate.shift(dist_y*RIGHT + x*4*IN))
                anims.append(VGroup(*tiles3[x][y]).animate.shift(y*4*IN + dist_x*DOWN))

                anims.append(VGroup(*tiles1[x+1][y]).animate.shift(dist_y*RIGHT + dist_x*DOWN))
                anims.append(VGroup(*tiles2[x+1][y]).animate.shift(dist_y*RIGHT + x*4*IN))
                anims.append(VGroup(*tiles3[x+1][y]).animate.shift(y*4*IN + dist_x*DOWN))

                anims.append(VGroup(*tiles1[x][y+1]).animate.shift(dist_y*RIGHT + dist_x*DOWN))
                anims.append(VGroup(*tiles2[x][y+1]).animate.shift(dist_y*RIGHT + x*4*IN))
                anims.append(VGroup(*tiles3[x][y+1]).animate.shift(y*4*IN + dist_x*DOWN))

                anims.append(VGroup(*tiles1[x+1][y+1]).animate.shift(dist_y*RIGHT + dist_x*DOWN))
                anims.append(VGroup(*tiles2[x+1][y+1]).animate.shift(dist_y*RIGHT + x*4*IN))
                anims.append(VGroup(*tiles3[x+1][y+1]).animate.shift(y*4*IN + dist_x*DOWN))

        with self.voiceover(text="""We can now divide this matrix into multiple hierarchicaly constructed tiles""") as trk:
            self.play(*anims)



