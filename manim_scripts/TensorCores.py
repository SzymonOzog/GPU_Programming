from manimlib import *
from math import radians

class TensorCores(Scene):
    def construct(self):
        # init scene
        total_r = 32
        start_r = 8
        self.play(*[FadeOut(x) for x in self.mobjects])
        mat1_f = [Square(stroke_width=0, fill_color=GREEN, fill_opacity=0.5) for _ in range(total_r*total_r)]
        mat2_f = [Square(stroke_width=0, fill_color=BLUE, fill_opacity=0.5) for _ in range(total_r*total_r)]
        mat3_f = [Square(stroke_width=0, fill_color=ORANGE, fill_opacity=0.5) for _ in range(total_r*total_r)]


        g1 = Group(*mat1_f).arrange_in_grid(total_r, total_r, buff=4).move_to(ORIGIN, aligned_edge=UL).shift(25*UP + 15*LEFT)
        g2 = Group(*mat2_f).arrange_in_grid(total_r, total_r, buff=4).next_to(g1, UP, buff = 2)
        g3 = Group(*mat3_f).arrange_in_grid(total_r, total_r, buff=4).next_to(g1, LEFT, buff = 2)

        self.frame.set_shape(183, 103)
        self.frame.move_to([-7, 24, 0])

        mat1 = []
        mat2 = []
        mat3 = []
        for i in range(start_r):
            for j in range(start_r):
                r = i
                c = j
                mat1.append(mat1_f[r*total_r + c])
                r = total_r - start_r + i
                c = j
                mat2.append(mat2_f[r*total_r + c])
                r = i
                c = total_r - start_r + j
                mat3.append(mat3_f[r*total_r + c])

        self.play(*[ShowCreation(x) for x in mat2])
        self.play(*[ShowCreation(x) for x in mat3])

        #highlight vectors
        v1 = [mat2[i*8] for i in range(start_r)]
        v2 = [mat3[i] for i in range(start_r)]

        self.play(*[v.animate.set_opacity(1) for v in v1])
        self.play(*[v.animate.set_opacity(1) for v in v2])

        #interm vector
        v3 = []
        for i in range(start_r):
            p1 = v2[i].get_center()
            p2 = v1[i].get_center()
            pos = [p1[0], p2[1], p2[2]] 
            v3.append(Square(stroke_width=0, fill_color=YELLOW, fill_opacity=1).move_to(pos))

        for i in range(start_r):
            vv1 = v1[i].copy()
            vv2 = v2[i].copy()
            self.play(vv1.animate.move_to(v3[i].get_center()),
                      vv2.animate.move_to(v3[i].get_center()))
            self.play(FadeOut(vv1), FadeOut(vv2), FadeIn(v3[i]))

        #accumulate
        def to_green(step):
            percentage_red = 1 - step/6
            red = int(255*percentage_red + 131 * (1-percentage_red)) 
            return f"#{red:02x}C167"

            
        for i in range(start_r-1):
            self.play(v3[i].animate.move_to(v3[i+1]),
                      v3[i+1].animate.set_color(to_green(i)))
            self.remove(v3[i])

        self.play(v3[-1].animate.move_to(mat1[0].get_center()))

        #move to 3d
        mat1_3d_f = [VCube(fill_color=GREY, fill_opacity=0.1).move_to(x.get_center()) for x in mat1_f]
        mat2_3d_f = [VCube(fill_color=x.get_color(), fill_opacity=0.3).move_to(x.get_center()).shift(2*IN) for x in mat2_f]
        mat3_3d_f = [VCube(fill_color=x.get_color(), fill_opacity=0.3).move_to(x.get_center()).shift(2*IN) for x in mat3_f]

        mat1_3d = []
        mat2_3d = []
        mat3_3d = []

        for i in range(start_r):
            for j in range(start_r):
                r = i
                c = j
                mat1_3d.append(mat1_3d_f[r*total_r + c])
                r = total_r - start_r + i
                c = j
                mat2_3d.append(mat2_3d_f[r*total_r + c])
                r = i
                c = total_r - start_r + j
                mat3_3d.append(mat3_3d_f[r*total_r + c])


        self.play(*[FadeOut(x)for x in mat1 + mat2 + mat3], 
                  *[FadeIn(x)for x in mat1_3d + mat2_3d + mat3_3d],
                  FadeOut(v3[-1]))

        #rotate matrices
        self.play(self.frame.animate.set_euler_angles(-2.24045432,  1.17009916,  1.86961547))

        mat2_3d_f_g = VGroup(*mat2_3d_f)
        mat3_3d_f_g = VGroup(*mat3_3d_f)

        mat2_3d_g = VGroup(*mat2_3d)
        mat3_3d_g = VGroup(*mat3_3d)
        self.play(mat2_3d_f_g.animate.rotate(radians(90), axis=LEFT, about_edge=DOWN), mat3_3d_f_g.animate.rotate(radians(90), axis=DOWN, about_edge=RIGHT))
        self.play(self.frame.animate.set_shape(137, 77).move_to([-8.3, 4.62, -16.36]))


        #show index calculation
        self.play(mat2_3d_f_g.animate.shift(4*IN).shift(4*UP), mat3_3d_f_g.animate.shift(4*IN).shift(4*LEFT))

        #highlight vectors
        frame_start = self.frame.copy()
        frame_end = self.frame.copy().set_euler_angles(0, 0, 0)
        frame_end.saved_alpha = 0
        def updater(m, dt):
            frame_end.saved_alpha = min(1, frame_end.saved_alpha+dt/50)
            m.interpolate(frame_start, frame_end, frame_end.saved_alpha)
            
        self.frame.add_updater(updater)

        for j in range(start_r):
            for k in range(start_r):
                run_time = 1 if j + k == 0 else 0.25
                v1 = [mat2_3d[i*8 + k] for i in range(8)]
                v2 = [mat3_3d[j*8 + i] for i in range(8)]

                disable_highlight = [
                        m.animate.set_opacity(0.3) for m in mat3_3d+mat2_3d if m not in v1+v2
                        ]
                
                self.play(*[v.animate.set_opacity(1) for v in v1], *[v.animate.set_opacity(1) for v in v2], *disable_highlight, run_time=run_time)

                dot_prod = []
                for i in range(8):
                    pos = mat1_3d[j*8 + k].get_center().copy()
                    pos[2] = v1[i].get_center()[2] - (43) + i * 6
                    dot_prod.append(VCube(fill_color=YELLOW).move_to(pos))

                anims = []
                for i in range(8):
                    c1 = v1[i].copy()
                    c2 = v2[i].copy()
                    dp = dot_prod[i]
                    anims.extend([ReplacementTransform(c1, dp),ReplacementTransform(c2, dp)])

                self.play(*anims, run_time=run_time)


                #sum dot products
                if j + k == 0:
                    run_time=0.5 if j + k == 0 else 0.03
                    for i in range(start_r-1):
                        tmp = dot_prod[i+1].copy().set_color(to_green(i))
                        self.play(Transform(dot_prod[i], tmp, run_time=run_time, rate_func=linear), Transform(dot_prod[i+1], tmp, run_time=run_time, rate_func=linear))
                        self.remove(dot_prod[i])

                    dot_prod[-1].deactivate_depth_test()
                    self.play(dot_prod[-1].animate(run_time=run_time, rate_func=linear).move_to(mat1_3d[j*8 + k].get_center()))
                else:
                    run_time = 0.25 if j < 2 else 0.1
                    dot_prod[-1].deactivate_depth_test()
                    tmp = dot_prod[-1].copy().set_color(to_green(7)).move_to(mat1_3d[j*8+k].get_center()).deactivate_depth_test()
                    self.play(*[Transform(x, tmp, run_time=run_time) for x in dot_prod])
        self.frame.remove_updater(updater)


        self.play(*[FadeIn(x) for x in mat1_3d_f + mat2_3d_f + mat3_3d_f])

