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

        ## create h(x) and differential

        ibp1 = MathTex(r"\int f(x)g'(x)dx =  f(x)g(x) - \int f'(x)g(x)dx", font_size = 60)
        h = MathTex(r"h(x) = f(x)g(x)", font_size = 70).shift(6*UP)
        hdash1 = MathTex(r"h'(x) = \frac{d}{dx}h(x) ", font_size = 70).shift(6*UP)
        hdash2 = MathTex(r"h'(x) = \frac{d}{dx}f(x)g(x)", font_size = 70).shift(6*UP)

        txt0 = Tex("Proof of Integration by Parts", font_size = 70).next_to(ibp1,UP*2)

        self.play(Write(ibp1), FadeIn(txt0))
        self.wait(1)
        self.play(FadeOut(ibp1),FadeOut(txt0))
        self.play(Write(h))
        self.wait(0.4)
        self.play(ReplacementTransform(h,hdash1))
        self.wait(0.5)
        self.play(ReplacementTransform(hdash1,hdash2))
        self.wait(1)


        ## using first principles for h(x) and subbing fg in

        txt1 = Tex("Using First Prinicples of h'(x)", font_size = 60).next_to(hdash2,DOWN)
        fp = MathTex(r"\lim_{\Delta x \to 0}\frac{h(x+\Delta x)-h(x)}{\Delta x}", font_size = 70).next_to(txt1,DOWN)
        fpfg = MathTex(r"= \lim_{\Delta x \to 0}\frac{f(x+\Delta x)g(x+\Delta x)-f(x)g(x)}{\Delta x}", font_size = 70).next_to(fp,DOWN)


        self.play(Write(txt1), FadeIn(fp))
        self.wait(1)
        self.play(FadeIn(fpfg))
        self.wait(1)


        ## adding and minusing f(x)g(x+dx) + rearraning

        fpfgpm0 = MathTex(r"= \lim_{\Delta x \to 0}\frac{f(x+\Delta x)g(x+\Delta x)-f(x)g(x)}{\Delta x} + 0", font_size = 60).next_to(fp,DOWN)
        fpfgpm1 = MathTex(r"= \lim_{\Delta x \to 0}\frac{f(x+\Delta x)g(x+\Delta x)-f(x)g(x)}{\Delta x}+ 1 - 1", font_size = 60).next_to(fp,DOWN)
        fpfgpm2 = MathTex(r"= \lim_{\Delta x \to 0}\frac{f(x+\Delta x)g(x+\Delta x)-f(x)g(x)}{\Delta x}  \\+ \frac{f(x)g(x+\Delta x)}{\Delta x}-\frac{f(x)g(x+\Delta x)}{\Delta x}", font_size = 70).next_to(fp,DOWN)
        fpfgpm3 = MathTex(r"= \lim_{\Delta x \to 0}\frac{f(x)(g(x+\Delta x)-g(x))+g(x+\Delta x)(f(x+\Delta x)-f(x))}{\Delta x}", font_size = 50).next_to(fp,DOWN)
        fpfgpm4 = MathTex(r"= \lim_{\Delta x \to 0}f(x) \frac{g(x+\Delta x)-g(x))}{\Delta x} \\+ \lim_{\Delta x \to 0}g(x+\Delta x) \frac{f(x+\Delta x)-f(x))}{\Delta x}", font_size = 70).next_to(fp,DOWN)
        fpfin = MathTex(r"= f(x)g'(x) + f'(x)g(x)", font_size = 70).next_to(fp,DOWN)

        self.play(ReplacementTransform(fpfg,fpfgpm0))
        self.wait(1)
        self.play(ReplacementTransform(fpfgpm0,fpfgpm1))
        self.wait(0.5)
        self.play(ReplacementTransform(fpfgpm1,fpfgpm2))
        self.wait(1)
        self.play(ReplacementTransform(fpfgpm2,fpfgpm3))
        self.wait(1)
        self.play(ReplacementTransform(fpfgpm3,fpfgpm4))
        self.wait(1)
        self.play(ReplacementTransform(fpfgpm4,fpfin))
        self.wait(1)

        ## equating to d/dxfg then integrating both sides

        prodrule = MathTex(r"\frac{d}{dx}f(x)g(x) = f(x)g'(x) + f'(x)g(x)", font_size = 70).next_to(fp,DOWN)
        txt2 = Tex("Integrating both sides", font_size = 60).next_to(prodrule,DOWN)
        intprodrule = MathTex(r"\int f(x)g'(x)dx + \int f'(x)g(x)dx = \int \frac{d}{dx}f(x)g(x)dx", font_size = 60).next_to(txt2,DOWN)

        self.play(ReplacementTransform(fpfin,prodrule))
        self.wait(1)
        self.play(FadeIn(txt2),Write(intprodrule))
        self.wait(1)

        ibppre = MathTex(r"\int f(x)g'(x)dx + \int f'(x)g(x)dx =  f(x)g(x) ", font_size = 60).next_to(txt2,DOWN)
        ibp = MathTex(r"\int f(x)g'(x)dx =  f(x)g(x) - \int f'(x)g(x)dx", font_size = 60).next_to(txt2,DOWN)
        txt3 = Tex("Integration by parts formula!", font_size = 80).next_to(ibp,DOWN)

        self.play(ReplacementTransform(intprodrule,ibppre))
        self.wait(1)
        self.play(ReplacementTransform(ibppre,ibp),Write(txt3))
        self.wait(1)



