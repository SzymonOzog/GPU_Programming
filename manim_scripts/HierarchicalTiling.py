
import os
from manimlib import *
from math import radians
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene

class HierarchicalTiling(VoiceoverScene):
    def construct(self):
        # init scene
        self.set_speech_service(
                RecorderService(transcription_model="base")
            # GTTSService(transcription_model="base")
            )
        self.voiceovers_in_embed = True

        total_n = 64
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
                dist_x = x*8 + 16*(x//4)
                dist_y = y*8 + 16*(y//4)
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


        #show warp distinction
        rects_b1 = []
        rects_b2 = []
        rects_b3 = []
        rects_b4 = []
        for x in range(0, n_tiles, 2):
            for y in range(0, n_tiles, 2):
                if x <= 2 and y <= 2:
                    rects = rects_b1
                elif x > 2 and y <= 2:
                    rects = rects_b2
                elif x <= 2 and y > 2:
                    rects = rects_b3
                else:
                    rects = rects_b4
                rects.append(SurroundingRectangle(VGroup(tiles1[x][y], tiles1[x+1][y], tiles1[x][y+1], tiles1[x+1][y+1]), buff=5))
        rects_t = [Text(f"Warp {i}", font_size=2000, base_color=YELLOW).move_to(r) for i, r in enumerate(rects_b1)]
        
        with self.voiceover(text="""In this setup each warp producess 4 output tiles""") as trk:
            self.play(self.frame.animate
                      .set_shape(1078.2404, 612.86053)
                      .move_to([145.04472, -178.0173, -15.1940775])
                      .set_euler_angles(0,0,0))
            self.play(*[ShowCreation(x) for x in rects_b1] + [Write(x) for x in rects_t])

        #show block distinction
        blocks = [SurroundingRectangle(VGroup(*r), color=RED, buff=8) for r in [rects_b1, rects_b3, rects_b2, rects_b4]]
        blocks_t = [Text(f"Block {i}", font_size=3000, fill_color=RED).move_to(r).shift(0.1*OUT) for i, r in enumerate(blocks)]
        with self.voiceover(text="""And each block produces 16 output tiles""") as trk:
            self.play(*[ShowCreation(x) for x in blocks] + [Write(x) for x in blocks_t])


        anims = []

        for x in range(n_tiles):
            for y in range(n_tiles):
                if x < 4 and y < 4:
                    anims.append(tiles1[x][y].animate.set_color(GREY))
        
        # for synchronizing with manimCE script
        timestamps = []
        def print_timestamp():
            timestamps.append(self.time)
            print(self.time)

        with self.voiceover(text="""We start similarly as before by zeroing out our accumulator""") as trk:
            self.play(*[FadeOut(x) for x in blocks + rects_t + blocks_t + rects_b1 + rects_b2 + rects_b3 + rects_b4],
                      self.frame.animate.set_shape(1160.2727, 659.4865)
                      .move_to([196.86966, -76.23208, 193.16118])
                      .set_euler_angles(-2.17929547,  0.88121027,  1.86961547))
            print_timestamp()
            self.play(*anims)

        # move to global memory
        anims = []
        for tt in tiles2:
            for t in tt:
                anims.append(t.animate.shift(100*UP))
        for tt in tiles3:
            for t in tt:
                anims.append(t.animate.shift(100*LEFT))
        with self.voiceover(text="""Our input tiles initially start out far away in a slow global memory""") as trk:
            self.play(*anims)

        #show shared memory load
        smem1 = []
        smem2 = []
        with self.voiceover(text="""We then load first parts of our tiles to shared memory""") as trk:
            anims = []
            print_timestamp()
            for i in range(4):
                smem1.append(tiles2[0][i].copy())
                smem2.append(tiles3[i][0].copy())
                anims.append(tiles2[0][i].animate.set_opacity(0.5))
                anims.append(tiles3[i][0].animate.set_opacity(0.5))
                anims.append(smem1[-1].animate.set_opacity(0.5).shift(60*DOWN))
                anims.append(smem2[-1].animate.set_opacity(0.5).shift(60*RIGHT))
            self.play(*anims)

        with self.voiceover(text="""And when loading our tiles, we can do that <bookmark mark='1'/> using 
                            faster vectorized loads""") as trk:
            self.wait_until_bookmark("1")
            print_timestamp()

        def crossing(m1, m2, m3):
            return np.array([m1.get_center()[0], m1.get_center()[1], m3.get_center()[2]])

        # load row 1 to registers
        anims = []
        reg1 = [[[] * 2 for i in range(2)] for j in range(2)]
        reg2 = [[None for i in range(2)] for j in range(2)]
        for r in range(2):
            for warp_m in range(2):
                for warp_n in range(2):
                    t1 = tiles1[warp_n*2][warp_m*2 + r]
                    t2 = tiles2[0][warp_m*2 + r]
                    t3 = tiles3[warp_n*2][0]
                    reg1[warp_m][warp_n].append(t2.copy())
                    if reg2[warp_m][warp_n] is None:
                        reg2[warp_m][warp_n] = t3.copy()
                    anims.append(reg1[warp_m][warp_n][-1].animate.align_to(t1, UP).shift(3*UP))
        with self.voiceover(text="""Then all warps in a block will load the data from the first input matrix
                            from shared memory to registers""") as trk:
            print_timestamp()
            self.play(*anims, self.frame.animate.set_euler_angles(-2.05697776,  0.77009916,  1.86961547))
 
        # load from b to reg 
        anims = []
        for warp_m in range(2):
            for warp_n in range(2):
                t1 = tiles1[warp_n*2][warp_m*2]
                t3 = reg2[warp_m][warp_n]
                anims.append(t3.animate.align_to(t1, LEFT).shift(3*LEFT))
        with self.voiceover(text="""Afterwards we can load one tile of our second matrix to registers from shared memory""") as trk:
            print_timestamp()
            self.play(*anims)

        #play mma
        anims1 = []
        anims2 = []
        for r in range(1):
            for warp_m in range(2):
                for warp_n in range(2):
                    t1 = tiles1[warp_n*2][warp_m*2 + r]
                    t2 = reg1[warp_m][warp_n][r]
                    t3 = reg2[warp_m][warp_n]
                    acc = VCube(side_length=w, fill_color=YELLOW, fill_opacity=0.3).move_to(crossing(t1, t2, t3))
                    anims1.append(ReplacementTransform(VGroup(t2.copy(), t3.copy()), acc))
                    tmp = t1.copy().set_opacity(1).set_color(to_green(1, 8))
                    anims2.extend([Transform(acc, tmp, remover=True), Transform(t1, tmp)])
        with self.voiceover(text="""We then perform the first""") as trk:
            print_timestamp()
            self.play(*anims1)
            self.play(*anims2)

        #play second mma
        anims1 = []
        anims2 = []
        for r in range(1, 2):
            for warp_m in range(2):
                for warp_n in range(2):
                    t1 = tiles1[warp_n*2][warp_m*2 + r]
                    t2 = reg1[warp_m][warp_n][r]
                    t3 = reg2[warp_m][warp_n]
                    acc = VCube(side_length=w, fill_color=YELLOW, fill_opacity=0.3).move_to(crossing(t1, t2, t3))
                    anims1.append(ReplacementTransform(VGroup(t2.copy(), t3.copy()), acc))
                    tmp = t1.copy().set_opacity(1).set_color(to_green(1, 8))
                    anims2.extend([Transform(acc, tmp, remover=True), Transform(t1, tmp)])
        with self.voiceover(text="""And then the second mma instruction on the tiles that were already in our registers""") as trk:
            print_timestamp()
            self.play(*anims1)
            self.play(*anims2)


        #show memory saves
        l1 = Line(tiles3[0][0], smem2[0])
        l2s = []
        for warp_m in range(2):
            for warp_n in range(2):
                l2s.append(Line(smem2[0], reg2[warp_m][warp_n].get_corner(UP)))
        l3s = []
        for warp_m in range(2):
            for warp_n in range(2):
                for r in range(2):
                    t1 = tiles1[warp_n*2][warp_m*2 + r]
                    l3s.append(Line(reg2[warp_m][warp_n], t1))
        with self.voiceover(text="""This is how hierarchical tiling speeds up our memory access, 
                            We only do one<bookmark mark='1'/> global memory access per tile,
                            that is then loaded 4 times<bookmark mark='2'/> from much faster shared memory
                            to registers that reuse it <bookmark mark='3'/>2 more times""") as trk:
            
            self.wait_until_bookmark("1")
            self.play(ShowCreation(l1))
            self.wait_until_bookmark("2")
            self.play(*[ShowCreation(l) for l in l2s])
            self.wait_until_bookmark("3")
            self.play(*[ShowCreation(l) for l in l3s])

        self.play(*[FadeOut(x) for x in [l1] + l2s + l3s])

        # play next row matmul
        with self.voiceover(text="""When all that is done we can load another tile of the second input matrix
                            from shared memory to fill next output tiles<bookmark mark='1'/> and perform the next series
                            of mma operations""") as trk:
            anims = []
            for warp_m in range(2):
                for warp_n in range(2):
                    anims.append(FadeOut(reg2[warp_m][warp_n]))
            self.play(*anims)

            anims = []
            for warp_m in range(2):
                for warp_n in range(2):
                    t1 = tiles1[warp_n*2 + 1][warp_m*2]
                    t3 = tiles3[warp_n*2 + 1][0].copy()
                    reg2[warp_m][warp_n] = t3
                    anims.append(t3.animate.align_to(t1, LEFT).shift(3*LEFT))
            print_timestamp()
            self.play(*anims)

            self.wait_until_bookmark("1")
            anims1 = []
            anims2 = []
            for r in range(2):
                for warp_m in range(2):
                    for warp_n in range(2):
                        t1 = tiles1[warp_n*2 + 1][warp_m*2 + r]
                        t2 = reg1[warp_m][warp_n][r]
                        t3 = reg2[warp_m][warp_n]
                        acc = VCube(side_length=w, fill_color=YELLOW, fill_opacity=0.3).move_to(crossing(t1, t2, t3))
                        anims1.append(ReplacementTransform(VGroup(t2.copy(), t3.copy()), acc))
                        tmp = t1.copy().set_opacity(1).set_color(to_green(1, 8))
                        anims2.extend([Transform(acc, tmp, remover=True), Transform(t1, tmp)])
            print_timestamp()
            self.play(*anims1)
            self.play(*anims2)

        # advance tile rows
        with self.voiceover(text="""When all that is done we can load another tile of the second input matrix
                            from shared memory""") as trk:
            anims = []
            tile = 0
            c = 0
            for warp_m in range(2):
                for warp_n in range(2):
                    anims.append(FadeOut(reg2[warp_m][warp_n]))
                    anims.extend([FadeOut(x) for x in reg1[warp_m][warp_n]])
            for i in range(4):
                anims.append(tiles2[tile*2 + c][i].animate.set_opacity(0.3))
                anims.append(tiles3[i][tile*2 + c].animate.set_opacity(0.3))
            print_timestamp()
            self.play(*anims, *[FadeOut(x) for x in smem1 + smem2])
            self.wait(1)

            c = 1
            smem1 = []
            smem2 = []
            anims = []
            for i in range(4):
                smem1.append(tiles2[tile*2 + c][i].copy())
                smem2.append(tiles3[i][tile*2 + c].copy())
                anims.append(tiles2[tile*2 + c][i].animate.set_opacity(0.5))
                anims.append(tiles3[i][tile*2 + c].animate.set_opacity(0.5))
                anims.append(smem1[-1].animate.set_opacity(0.5).shift(60*DOWN))
                anims.append(smem2[-1].animate.set_opacity(0.5).shift(60*RIGHT))
            print_timestamp()
            self.play(*anims)

        with self.voiceover(text="""And we continue as before loading from shared memory into registers""") as trk:
            anims = []
            reg1 = [[[] * 2 for i in range(2)] for j in range(2)]
            for r in range(2):
                for warp_m in range(2):
                    for warp_n in range(2):
                        t1 = tiles1[warp_n*2][warp_m*2 + r]
                        t2 = tiles2[c][warp_m*2 + r]
                        reg1[warp_m][warp_n].append(t2.copy())
                        anims.append(reg1[warp_m][warp_n][-1].animate.align_to(t1, UP).shift(3*UP))
            print_timestamp()
            self.play(*anims)

        #next part
        with self.voiceover(text="""while reusing the data as much as possible for our MMA instructions""") as trk:
            anims1 = []
            anims2 = []
            anims3 = []
            anims4 = []
            for r in range(2):
                for warp_m in range(2):
                    for warp_n in range(2):
                        t1 = tiles1[warp_n*2 + r][warp_m*2]
                        t3 = tiles3[warp_n*2 + r][c].copy()
                        reg2[warp_m][warp_n] = t3
                        anims1.append(t3.animate.align_to(t1, LEFT).shift(3*LEFT))

                for k in range(2):
                    for warp_m in range(2):
                        for warp_n in range(2):
                            t1 = tiles1[warp_n*2 + r][warp_m*2 + k]
                            t2 = reg1[warp_m][warp_n][k]
                            t3 = reg2[warp_m][warp_n]
                            acc = VCube(side_length=w, fill_color=YELLOW, fill_opacity=0.3).move_to(crossing(t1, t2, t3))
                            anims2.append(ReplacementTransform(VGroup(t2.copy(), t3.copy()), acc))
                            tmp = t1.copy().set_opacity(1).set_color(to_green(tile*2+c, 8))
                            anims3.extend([Transform(acc, tmp, remover=True), Transform(t1, tmp)])
                for warp_m in range(2):
                    for warp_n in range(2):
                        anims4.append(FadeOut(reg2[warp_m][warp_n]))
            self.play(*anims1)
            print_timestamp()
            self.play(*anims2)
            self.play(*anims3)
            self.play(*anims4)
        anims = []
        for warp_m in range(2):
            for warp_n in range(2):
                anims.extend([FadeOut(x) for x in reg1[warp_m][warp_n]])
        for i in range(4):
            anims.append(tiles2[tile*2 + c][i].animate.set_opacity(0.3))
            anims.append(tiles3[i][tile*2 + c].animate.set_opacity(0.3))
        self.play(*anims, *[FadeOut(x) for x in smem1 + smem2])

        #continue untill the end of the matmul
        with self.voiceover(text="""We can continue with this pattern until we reach the end of our input matrices,
                            also the 2 by 2 tile sizes that I've shown here are not exactly indicative of what our real world tile sizes
                            will be, for benchmarking it I've been running for different configurations and saving the fastest one.
                            As always all code is available on my github that is linked in the description""") as trk:
            for tile in range(1, 4):
                for c in range(2):
                    smem1 = []
                    smem2 = []
                    anims = []
                    for i in range(4):
                        smem1.append(tiles2[tile*2 + c][i].copy())
                        smem2.append(tiles3[i][tile*2 + c].copy())
                        anims.append(tiles2[tile*2 + c][i].animate.set_opacity(0.5))
                        anims.append(tiles3[i][tile*2 + c].animate.set_opacity(0.5))
                        anims.append(smem1[-1].animate.set_opacity(0.5).shift(60*DOWN))
                        anims.append(smem2[-1].animate.set_opacity(0.5).shift(60*RIGHT))
                    print_timestamp()
                    self.play(*anims)

                    anims = []
                    reg1 = [[[] * 2 for i in range(2)] for j in range(2)]
                    for r in range(2):
                        for warp_m in range(2):
                            for warp_n in range(2):
                                t1 = tiles1[warp_n*2][warp_m*2 + r]
                                t2 = tiles2[tile*2 + c][warp_m*2 + r]
                                reg1[warp_m][warp_n].append(t2.copy())
                                anims.append(reg1[warp_m][warp_n][-1].animate.align_to(t1, UP).shift(3*UP))
                    print_timestamp()
                    self.play(*anims)

                    anims1 = []
                    anims2 = []
                    anims3 = []
                    anims4 = []
                    for r in range(2):
                        anims = []
                        for warp_m in range(2):
                            for warp_n in range(2):
                                t1 = tiles1[warp_n*2 + r][warp_m*2]
                                t3 = tiles3[warp_n*2 + r][tile*2 + c].copy()
                                reg2[warp_m][warp_n] = t3
                                anims1.append(t3.animate.align_to(t1, LEFT).shift(3*LEFT))

                        for k in range(2):
                            for warp_m in range(2):
                                for warp_n in range(2):
                                    t1 = tiles1[warp_n*2 + r][warp_m*2 + k]
                                    t2 = reg1[warp_m][warp_n][k]
                                    t3 = reg2[warp_m][warp_n]
                                    acc = VCube(side_length=w, fill_color=YELLOW, fill_opacity=0.3).move_to(crossing(t1, t2, t3))
                                    anims2.append(ReplacementTransform(VGroup(t2.copy(), t3.copy()), acc))
                                    tmp = t1.copy().set_opacity(1).set_color(to_green(tile*2+c, 8))
                                    anims3.extend([Transform(acc, tmp, remover=True), Transform(t1, tmp)])

                        for warp_m in range(2):
                            for warp_n in range(2):
                                anims4.append(FadeOut(reg2[warp_m][warp_n]))
                    print_timestamp()
                    self.play(*anims1)
                    self.play(*anims2)
                    self.play(*anims3)
                    self.play(*anims4)
                    anims = []
                    for warp_m in range(2):
                        for warp_n in range(2):
                            anims.extend([FadeOut(x) for x in reg1[warp_m][warp_n]])
                    for i in range(4):
                        anims.append(tiles2[tile*2 + c][i].animate.set_opacity(0.3))
                        anims.append(tiles3[i][tile*2 + c].animate.set_opacity(0.3))
                    self.play(*anims, *[FadeOut(x) for x in smem1 + smem2])

        print_timestamp()
        self.play(*[FadeOut(x) for x in self.mobjects])
        print("timestamps = ", timestamps)  
