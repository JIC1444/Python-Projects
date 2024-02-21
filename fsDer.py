from manim import*


SCALE_FACTOR = 1
tmp_pixel_height = config.pixel_height
config.pixel_height = config.pixel_width
config.pixel_width = tmp_pixel_height

config.frame_height = config.frame_height / SCALE_FACTOR
config.fram_width = config.frame_height * 9/16
FRAME_HEIGHT = config.frame_height
FRAME_WIDTH = config.frame_width


class PhoneBorder(Scene):
    def setup(self, add_border=True):
        if add_border:
            self.border = Rectangle(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            color = WHITE
        )
        self.add(self.border)

class Proof(Scene):
    def construct(self): 

        fs = MathTex(r"f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty (a_n cos( \frac{\pi nx}{L} + b_n sin( \frac{\pi nx}{L}) ", font_size=60)

        txt0 = Tex("Derivation of the Fourier Series",font_size = 70).shift(4*UP)

        self.play(Write(fs),FadeIn(txt0))
        self.play(Unwrite(fs),FadeOut(txt0))

        txt11=Text("f(x) must follow the Dirichlet Conditions: ", font_size = 50).shift(7*UP)
        txt12 =Text("-Be periodic with a ",font_size = 50).shift(4*UP,2.9*LEFT)
        txt121 = Text("finite number of ",color=TEAL, font_size=50).next_to(txt12,RIGHT)
        txt1211 = Text("discontinuities ",color=TEAL, font_size=50).next_to(txt12,DOWN)
        txt122 = Text(" within one period", font_size=50).next_to(txt1211,RIGHT)
        txt13 = Text("-Absolute value of f(x) must",font_size = 50).shift(1*UP,1.3*LEFT)
        txt131 =Text(" converge",font_size = 50,color=TEAL).next_to(txt13,RIGHT)
        txt14 = Text("-And/or a finite number of ",font_size =50).shift(2*DOWN,1.5*LEFT)
        txt141 = Text("minima or ",font_size =50, color=TEAL).next_to(txt14,RIGHT)
        txt1411 = Text("maxima ",font_size =50, color=TEAL).next_to(txt14,DOWN)
        txt142= Text("within one period",font_size =50).next_to(txt1411,RIGHT)
        txt15 = Text("So let us assume our f(x) \nfollows these conditions",font_size = 50, color=PURPLE).shift(5*DOWN)



        self.play(FadeIn(txt11))
        self.wait(0.5)
        self.play(FadeIn(txt12),FadeIn(txt121),FadeIn(txt122),FadeIn(txt1211))
        self.wait(0.5)
        self.play(FadeIn(txt13),FadeIn(txt131))
        self.wait(0.5)
        self.play(FadeIn(txt14),FadeIn(txt141),FadeIn(txt142),FadeIn(txt1411))
        self.wait(0.5)
        self.play(FadeIn(txt15))
        self.wait(1)
        self.play(FadeOut(txt11),FadeOut(txt12),FadeOut(txt121),FadeOut(txt1211),FadeOut(txt122),FadeOut(txt13),FadeOut(txt131),FadeOut(txt14),FadeOut(txt141),FadeOut(txt1411),FadeOut(txt142),FadeOut(txt15))
        self.wait(0.5)


        ## a0 special case:

        ooga = MathTex(r"f(x) = \sum_{n=0}^\infty a_n cos( \frac{n \pi x}{L}) + b_n sin( \frac{n \pi x}{L})")
        txt9=Text("As f(x) follows the Dirichlet Conditions,\nwe can write f(x) as a function of sines and cosines: ",font_size = 55).next_to(ooga,UP)
        txt10=Text("However there is a special case at n=0")
        oogasp = MathTex(r"u_0 = a_0 cos(0) + b_0 sin(0)")
        oogasp1 = MathTex(r"u_0 = 1 a_0  + 0")
        oogasp2 = MathTex(r"u_0 = a_0")
        ooga2 = MathTex(r"f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty a_n cos( \frac{n \pi x}{L}) + b_n sin( \frac{n \pi x}{L})")
        ooga2up = MathTex(r"f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty a_n cos( \frac{n \pi x}{L}) + b_n sin( \frac{n \pi x}{L})").shift(11*UP)
        txt11=Text("We will then write a_0/2 for ease later").next_to(ooga,UP)        
        boxup = SurroundingRectangle(ooga2up, color=YELLOW, buff=MED_LARGE_BUFF)
        mobjects = VGroup(VGroup(boxup, ooga2up)).arrange(DOWN)


        self.play(FadeIn(txt9),Write(ooga))
        self.play(FadeIn(txt10))
        self.wait(0.5)
        self.play(FadeOut(txt9),FadeOut(txt10))
        self.play(ReplacementTransform(ooga,oogasp))
        self.wait(0.5)
        self.play(ReplacementTransform(oogasp,oogasp1))
        self.play(ReplacementTransform(oogasp1,oogasp2))
        self.wait(0.5)
        self.play(FadeOut(oogasp2))
        self.play(Write(ooga2), FadeIn(txt11))
        self.wait(0.5)
        ##put in box??
        self.play(FadeOut(ooga2),FadeOut(txt11),Write(ooga2up),Create(mobjects))




        ## by exploiting the orthogonality of sine and cosine, we will create some useful equations which will be used later


        orth1 =MathTex(r" \int_{-L}^L sin( \frac{m \pi x}{L}) sin( \frac{n \pi x}{L})dx", font_size = 70)
        
        rel1 = MathTex(r"sin(m)sin(n) = \frac{1}{2} (cos(m-n) - cos(m+n))", font_size = 65)
        txt2 = Text("Using relation: ", font_size = 60).next_to(rel1,2*UP)
        orth12 = MathTex(r"= \frac{1}{2} \int_{-L}^L cos( \frac{(m-n) \pi x}{L}) - cos( \frac{(m+n) \pi x}{L})dx",font_size=70)
        orth13 = MathTex(r"= \frac{1}{2} [ \frac{L}{\pi (m-n)} sin( \frac{(m-n) \pi x}{L})\\ - \frac{L}{\pi (m+n)}sin( \frac{(m+n) \pi x}{L})]_{-L}^L",font_size=70)
        orth131 = MathTex(r"= \frac{1}{2} [ \frac{L}{\pi (m-n)} sin( \frac{(m-n) \pi x}{L})\\ - \frac{L}{\pi (m+n)}sin( \frac{(m+n) \pi x}{L})]_{-L}^L",font_size=70)
        orth000 = MathTex(r"= \frac{1}{2} [ \frac{L}{\pi (m-n)} sin( \frac{(m-n) \pi L}{L})\\ - \frac{L}{\pi (m+n)}sin( \frac{(m+n) \pi L}{L})] \\- \frac{1}{2} [ \frac{L}{\pi (m-n)} sin( \frac{(m-n) \pi (-L)}{L}) \\- \frac{L}{\pi (m+n)}sin( \frac{(m+n) \pi (-L)}{L})] ",font_size=70)
        orth00 = MathTex(r"= \frac{1}{2} [ \frac{L}{\pi (m-n)} sin((m-n) \pi)\\ - \frac{L}{\pi (m+n)}sin((m+n) \pi)] \\- \frac{1}{2} [ \frac{L}{\pi (m-n)} sin(-(m-n) \pi)\\ - \frac{L}{\pi (m+n)}sin(-(m+n) \pi)] ",font_size=70)
        orth0 = MathTex(r"= \frac{1}{2} [ \frac{L}{\pi (m-n)} sin(0)\\ - \frac{L}{\pi (m+n)}sin(0]",font_size=70)
        orth01 = MathTex(r"= \frac{1}{2} [ 0 - 0]",font_size=70)
        orth02 = MathTex(r"\int_{-L}^L sin( \frac{m \pi x}{L}) sin( \frac{n \pi x}{L})dx = 0",font_size=70)
        txt4 = Tex("if m ",font_size=90).shift(6*UP)
        txt41 = MathTex("\neq ",font_size=90).next_to(txt4,RIGHT)
        txt42 = Tex(" n ",font_size=90).next_to(txt41,RIGHT)

        self.play(Write(orth1))
        self.play(ReplacementTransform(orth1,rel1),FadeIn(txt2))
        self.wait(1)
        self.play(FadeOut(txt2))
        self.play(ReplacementTransform(rel1,orth12))
        self.wait(0.5)
        self.play(ReplacementTransform(orth12,orth13))
        self.wait(0.5)
        self.play(ReplacementTransform(orth13,orth131))
        self.wait(0.5)
        self.play(ReplacementTransform(orth131,orth000),Write(txt4),Write(txt41),Write(txt42))
        self.wait(0.5)
        self.play(ReplacementTransform(orth000,orth00))
        self.wait(0.5)
        self.play(ReplacementTransform(orth00,orth0))
        self.wait(0.5)
        self.play(ReplacementTransform(orth0,orth01))
        self.wait(0.5)
        self.play(ReplacementTransform(orth01,orth02))
        self.wait(0.5)
        self.play(FadeOut(orth02),Unwrite(txt4),Unwrite(txt41),Unwrite(txt42))
        self.wait(0.5)

        rel2 = MathTex(r"sin(m)sin(m) = 1 - cos(2mx)")
        txt5 = Tex("Using relation: ",font_size=60).next_to(rel2,2*UP)
        orthL0 = MathTex(r"\int_{-L}^L sin( \frac{m \pi x}{L}) sin( \frac{n \pi x}{L})dx \\= \int_{-L}^L (1 - cos(2m \pi x) dx",font_size=70)
        orthL00 = MathTex(r"= \frac{1}{2} [x - \frac{L}{2m \pi } sin( \frac{2m \pi x}{L})]_{-L}^L",font_size=70)
        orthL01 = MathTex(r"= \frac{1}{2} [L - \frac{L}{2m \pi } sin( \frac{2m \pi L}{L})]\\ - [-L - \frac{L}{2m \pi } sin( \frac{2m \pi (-L)}{L})]",font_size=70)
        orthL02 = MathTex(r"= \frac{1}{2} [L - \frac{L}{2m \pi } sin(2m \pi )]\\ - [-L - \frac{L}{2m \pi } sin(-2m \pi )]",font_size=70)
        orthL03 = MathTex(r"= \frac{1}{2} [L - \frac{L}{2m \pi } 0 - [-L - \frac{L}{2m \pi } 0]",font_size=70)
        orthL04 = MathTex(r"= \frac{1}{2} [L + L]",font_size=70)
        orthL05 = MathTex(r"= \frac{1}{2} 2L",font_size=70)
        orthL06 = MathTex(r"= L",font_size=70)
        orthL07 = MathTex(r" \int_{-L}^L sin( \frac{m \pi x}{L}) sin( \frac{n \pi x}{L})dx = L",font_size=70)
        txt6 = Tex("if m = n", font_size = 65).shift(6*UP)


        self.play(Write(orthL0),Write(txt6))
        self.wait(0.5)
        self.play(ReplacementTransform(orthL0,orthL00))
        self.wait(0.5)
        self.play(ReplacementTransform(orthL00,orthL01))
        self.wait(0.5)
        self.play(ReplacementTransform(orthL01,orthL02))
        self.wait(0.5)
        self.play(ReplacementTransform(orthL02,orthL03))
        self.wait(0.5)
        self.play(ReplacementTransform(orthL03,orthL04))
        self.wait(0.5)
        self.play(ReplacementTransform(orthL04,orthL05))
        self.wait(0.3)
        self.play(ReplacementTransform(orthL05,orthL06))
        self.wait(0.3)
        self.play(FadeOut(orthL06))
        self.play(Write(orthL07))
        self.wait(0.5)
        self.play(FadeOut(orthL07),FadeOut(txt6))
        self.wait(0.5)

        ## establish kronecker delta
        kronecker = MathTex(r"L \delta_{mn} = ",font_size=80)
        txt3= Text("Using the Kronecker Delta to \nsimplify the two cases into one equation",font_size=55).next_to(kronecker,UP*3)
        
        orth14 = Tex(r" L ",r"  if m = n\\",font_size=70).next_to(kronecker,DOWN)
        orth142 = Tex(r" 0 ",font_size=70).next_to(orth14[0],DOWN)
        orth143 = Tex(r" if",font_size=70).next_to(orth142,RIGHT)
        orth141 = MathTex(r"m \neq n",font_size=70).next_to(orth143,RIGHT)


        self.play(FadeIn(txt3))
        self.wait(0.5)
        self.play(Write(kronecker),FadeIn(orth14[0]),FadeIn(orth14[1]),FadeIn(orth143),FadeIn(orth142),FadeIn(orth141))
        self.wait(0.5)
        self.play(FadeOut(orth14[0]),FadeOut(orth14[1]),FadeOut(orth143),FadeOut(orth142),FadeOut(orth141),Unwrite(kronecker),FadeOut(txt3))
        self.wait(0.5)



        ## state the rule for cos


        orth2 =MathTex(r" \int_{-L}^L cos( \frac{m \pi x}{L}) cos( \frac{n \pi x}{L})dx = L \delta_{mn} ", font_size = 60)
        txt4 = Tex("The same result is true for cosine",font_size=55).next_to(orth2,UP)
        self.play(FadeIn(txt4),Write(orth2))
        self.wait(0.5)
        self.play(FadeOut(orth2),FadeOut(txt4))
        self.wait(0.5)


        ## do sincos integral

        sci = MathTex(r"\int_{-L}^L sin( \frac{m \pi x}{L}) cos( \frac{n \pi x}{L})dx ", font_size = 70)
        txt8=Text("The final integral involving sine and cosine",font_size=55).shift(3*UP)
        txt7 = Text("Which is an odd function,\n its integral is always 0",font_size=55).shift(3*DOWN)
        sci0 = MathTex(r"\int_{-L}^L sin( \frac{m \pi x}{L}) cos( \frac{n \pi x}{L})dx = 0",font_size=70)

        self.play(Write(sci),FadeIn(txt8))
        self.wait(0.5)
        self.play(FadeIn(txt7))
        self.wait(0.5)
        self.play(ReplacementTransform(sci,sci0))
        self.wait(0.5)
        self.play(FadeOut(txt7),FadeOut(sci0),FadeOut(txt8))
        self.wait(0.5)



        ##write all equations up the top for later use

        orth1top =MathTex(r" \int_{-L}^L sin( \frac{m \pi x}{L}) sin( \frac{n \pi x}{L})dx = L \delta_{mn} ", font_size = 50).shift(10*UP)
        orth2top =MathTex(r"\int_{-L}^L cos( \frac{m \pi x}{L}) cos( \frac{n \pi x}{L})dx = L \delta_{mn}",font_size = 50).next_to(orth1top,DOWN)
        orth3top = MathTex(r"\int_{-L}^L sin( \frac{m \pi x}{L}) cos( \frac{n \pi x}{L})dx = 0",font_size=50).next_to(orth2top,DOWN)
        ## put box around them
        
        self.play(FadeIn(orth1top),FadeIn(orth2top),FadeIn(orth3top))
        self.wait(0.5)



        ## finding a_0

        a00 =  MathTex(r" \int_{-L}^L f(x) dx ", font_size = 60)
        txt15 = Text("Finding a0").next_to(a00,UP)
        a01 = MathTex(r" \int_{-L}^L \frac{a_0}{2} dx \\+ \int_{-L}^L \sum_{n=1}^\infity [a_n cos( \frac{n \pi x}{L}) \\+ b_n sin( \frac{n \pi x}{L})] dx ",font_size=60)
        a02 = MathTex(r" \int_{-L}^L \frac{a_0}{2} dx + 0 + 0 ",font_size=60)
        a03 = MathTex(r" [ \frac{a_0 x}{2}]_{-L}^L ",font_size=60)
        a04 = MathTex(r" \frac{1}{2} (La_0 + La_0)")
        a05 = MathTex(r"La_0")

        coef1 = MathTex(r"a_0 = \frac{1}{L} \int_{-L}^L f(x) dx ")

        self.play(Write(a00),FadeIn(txt15))
        self.wait(0.5)
        self.play(ReplacementTransform(a00,a01),FadeOut(txt15))
        self.wait(0.5)
        self.play(ReplacementTransform(a01,a02))
        self.wait(0.5)
        self.play(ReplacementTransform(a02,a03))
        self.wait(0.5)
        self.play(ReplacementTransform(a03,a04))
        self.wait(0.5)
        self.play(ReplacementTransform(a04,a05))
        self.wait(0.5)
        self.play(FadeOut(a05))
        self.play(Write(coef1))
        self.wait(0.5)
        self.play(FadeOut(coef1))

        ## put in box below?
 
 
 
       ## finding a_n

        an00 = MathTex(r"\int_{-L}^L f(x)cos( \frac{m \pi x}{L}) dx ", font_size = 60)
        txt16 = Text("Finding a_n",font_size = 50).next_to(an00,UP)
        an01 = MathTex(r"\int_{-L}^L a_0 cos( \frac{m \pi x}{L}) dx", r"\\+ \sum_{n=1}^\infty [\int_{-L}^L a_n cos({m \pi x}{L})cos( \frac{n \pi x}{L}) dx \sum_{n=1}^\infty",r"\\ + \sum_{n=1}^\infty [\int_{-L}^L b_n sin({n \pi x}{L})cos( \frac{m \pi x}{L}) dx]", font_size = 60)
        an02 = MathTex(r"0",r" \sum_{n=0}^\infty a_n L \delta_(mn)",r"0")
        ## add brace saying which orth rule used
        an03 = MathTex(r"\sum_{n=0}^\infty a_n L")
        an04 = MathTex(r"a_m L")

        coef2 = MathTex(r"a_m = \frac{1}{L} \int_{-L}^L f(x)cos( \frac{m \pi x}{L})dx")



        self.play(Write(an00),FadeIn(txt16))

        self.play(FadeOut(txt16))
        self.wait(0.5)
        self.play(ReplacementTransform(an00,an01))
        self.wait(0.5)
        self.play(ReplacementTransform(an01,an02))
        self.wait(0.5)
        self.play(ReplacementTransform(an02,an03))
        self.wait(0.5)
        self.play(ReplacementTransform(an03,an04))
        self.wait(0.5)
        self.play(ReplacementTransform(an04,coef2))
        self.wait(0.5)
        self.play(Unwrite(coef2))
        ##box it below

        ## finding b_n

        bn00 = MathTex(r"\int_{-L}^L f(x)sin( \frac{m \pi x}{L}) dx ", font_size = 60)
        txt17 = Text("Finding b_n",font_size = 50).next_to(bn00,UP)
        bn01 = MathTex(r"\int_{-L}^L a_0 sin( \frac{m \pi x}{L}) dx", r"\\+ \sum_{n=1}^\infty [\int_{-L}^L a_n cos({m \pi x}{L})sin( \frac{n \pi x}{L}) dx \sum_{n=1}^\infty",r"\\ + \sum_{n=1}^\infty [\int_{-L}^L b_n sin({n \pi x}{L})sin( \frac{m \pi x}{L}) dx]", font_size = 60)
        bn02 = MathTex(r"0",r"+ 0",r"b_n L \delta_(mn)")
        ## add brace saying which orth rule used
        bn03 = MathTex(r"b_m L")

        coef3 = MathTex(r"b_m = \frac{1}{L} \int_{-L}^L f(x)sin( \frac{m \pi x}{L})dx")

        self.play(Write(bn00),FadeIn(txt17))
        self.wait(0.5)
        self.play(FadeOut(txt17))

        self.play(ReplacementTransform(bn00,bn01))
        self.wait(0.5)
        self.play(ReplacementTransform(bn01,bn02))
        self.wait(0.5)
        self.play(ReplacementTransform(bn02,bn03))
        self.wait(0.5)
        self.play(ReplacementTransform(bn03,coef3))
        self.wait(0.5)
        self.play(Unwrite(coef3))
        ##box it below




        close = Text("We have now derived all elements of a Fourier Series!")






