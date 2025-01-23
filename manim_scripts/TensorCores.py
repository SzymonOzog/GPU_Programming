from manimlib import *
from math import radians

class TensorCores(Scene):
    def construct(self):
        # init scene
        self.play(*[FadeOut(x) for x in self.mobjects])
        mat1 = [Square(stroke_width=0, fill_color=GREEN, fill_opacity=0.5) for _ in range(64)]
        mat2 = [Square(stroke_width=0, fill_color=BLUE, fill_opacity=0.5) for _ in range(64)]
        mat3 = [Square(stroke_width=0, fill_color=ORANGE, fill_opacity=0.5) for _ in range(64)]

        self.frame.set_shape(183, 103)
        self.frame.move_to([-7, 24, 0])

        g1 = Group(*mat1).arrange_in_grid(8, 8, buff=4)
        g2 = Group(*mat2).arrange_in_grid(8, 8, buff=4).next_to(g1, UP, buff = 2)
        self.play(*[ShowCreation(x) for x in mat2])

        g3 = Group(*mat3).arrange_in_grid(8, 8, buff=4).next_to(g1, LEFT, buff = 2)
        self.play(*[ShowCreation(x) for x in mat3])

        #highlight vectors
        v1 = [mat2[i*8] for i in range(8)]
        v2 = [mat3[i] for i in range(8)]

        self.play(*[v.animate.set_opacity(1) for v in v1])
        self.play(*[v.animate.set_opacity(1) for v in v2])

        #interm vector
        v3 = []
        for i in range(8):
            p1 = v2[i].get_center()
            p2 = v1[i].get_center()
            pos = [p1[0], p2[1], p2[2]] 
            v3.append(Square(stroke_width=0, fill_color=YELLOW, fill_opacity=1).move_to(pos))

        for i in range(8):
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

            
        for i in range(7):
            self.play(v3[i].animate.move_to(v3[i+1]),
                      v3[i+1].animate.set_color(to_green(i)))
            self.remove(v3[i])

        self.play(v3[-1].animate.move_to(mat1[0].get_center()))

        #move to 3d
        mat1_3d = [Cube(color=x.get_color()).move_to(x.get_center()) for x in mat1]
        mat2_3d = [Cube(color=x.get_color()).move_to(x.get_center()).shift(2*IN) for x in mat2]
        mat3_3d = [Cube(color=x.get_color()).move_to(x.get_center()).shift(2*IN) for x in mat3]

        self.play(*[FadeOut(x)for x in mat1 + mat2 + mat3], 
                  *[FadeIn(x)for x in mat1_3d + mat2_3d + mat3_3d])


        #rotate matrices
        self.play(self.frame.animate.set_euler_angles(-2.19045432,  1.22009916,  1.86961547))
        mat2_3d_g = Group(*mat2_3d)
        mat3_3d_g = Group(*mat3_3d)
        self.play(mat2_3d_g.animate.rotate(radians(90), axis=LEFT, about_edge=DOWN), mat3_3d_g.animate.rotate(radians(90), axis=DOWN, about_edge=RIGHT))
        self.play(self.frame.animate.set_shape(123, 69).move_to([-5.3, 2.97, -9.36]))

        #show index calculation
        mat2_3d_g.shift(4*IN).shift(4*UP)
        mat3_3d_g.shift(4*IN).shift(4*LEFT)

        dot_prod = []
        for i in range(8):
            pos = mat1_3d[0].get_center().copy()
            pos[2] = mat2_3d[i*8].get_center()[2]
            dot_prod.append(Cube(color=YELLOW).move_to(pos))

        anims = []
        for i in range(8):
            c1 = mat3_3d[i].copy()
            c2 = mat2_3d[i*8].copy()
            dp = dot_prod[i]
            anims.extend([ReplacementTransform(c1, dp),ReplacementTransform(c2, dp)])

        self.play(*anims)


        #sum dot products
        for i in range(7):
            tmp = dot_prod[i+1].copy().set_color(to_green(i))
            self.play(Transform(dot_prod[i], tmp), Transform(dot_prod[i+1], tmp))
            self.remove(dot_prod[i])

        self.play(dot_prod[-1].animate.move_to(mat1_3d[0].get_center()))
