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

        

        e= MathTex(r"e^{i\theta} = Cos(\theta) + iSin(\theta)",
                        font_size = 100  )

        eMacSeries = MathTex(r"e^{i\theta} = \sum_{n=0}^\infty \frac{(i\theta)^n}{n!} ",
                color= WHITE,
                font_size=100)
        
        MacSeriesExp = MathTex(r"\sum_{n=0}^\infty \frac{(i\theta)^n}{n!}",r"=",r" \frac{(i\theta)^0}{0!}",r"+ \frac{(i\theta)^1}{1!}",r"+ \frac{(i\theta)^2}{2!} ",r"\\+ \frac{(i\theta)^3}{3!}",r"+ \frac{(i\theta)^4}{4!}",r"+ \frac{(i\theta)^5}{5!} + ... " ,
                        font_size = 90   )

        txt0 = Tex("Proof of Euler's Formula", font_size = 100).next_to(e,UP)

        ## write e to i0 then make equal to maclaurin series for e to x


        self.play(Write(e),FadeIn(txt0))
        self.wait(0.2)
        self.play(ReplacementTransform(e,eMacSeries),FadeOut(txt0))
        self.wait(0.5)
        
        

        ## make mac series equal to expansion of raw terms

        txt1 = Tex("Using the Maclaurin Series of e", font_size = 90).next_to(MacSeriesExp,UP)
        
        self.play(ReplacementTransform(eMacSeries,MacSeriesExp), FadeIn(txt1))
        self.wait(0.25)
        self.play(FadeOut(txt1))


        ## simplify terms one by one by using transform

        MacSeriesExp1 = MathTex(r"\sum_{n=0}^\infty \frac{(i\theta)^n}{n!}",r"=",r" 1",r"+ \frac{(i\theta)^1}{1!}",r"+ \frac{(i\theta)^2}{2!} ",r"\\+ \frac{(i\theta)^3}{3!}",r"+ \frac{(i\theta)^4}{4!}",r"+ \frac{(i\theta)^5}{5!} + ... ",
                font_size = 90  )
        MacSeriesExp2 = MathTex(r"\sum_{n=0}^\infty \frac{(i\theta)^n}{n!}",r"=",r" 1",r"+ i\theta",r"+ \frac{(i\theta)^2}{2!}  ",r"\\+ \frac{(i\theta)^3}{3!}",r"+ \frac{(i\theta)^4}{4!}",r"+ \frac{(i\theta)^5}{5!} + ... "  ,
                    font_size = 90  )
        MacSeriesExp3 = MathTex(r"\sum_{n=0}^\infty \frac{(i\theta)^n}{n!}",r"=",r" 1",r"+ i\theta",r"- \frac{(\theta)^2}{2!}  ",r"\\+ \frac{(i\theta)^3}{3!}",r"+ \frac{(i\theta)^4}{4!}",r"+ \frac{(i\theta)^5}{5!} + ... " ,
                    font_size = 90   )
        MacSeriesExp4 = MathTex(r"\sum_{n=0}^\infty \frac{(i\theta)^n}{n!}",r"=",r"1",r"+ i\theta",r"- \frac{(\theta)^2}{2!}  ",r"\\- \frac{i(\theta)^3}{3!}",r"+ \frac{(i\theta)^4}{4!}",r"+ \frac{(i\theta)^5}{5!} + ... ",
                     font_size = 90    )
        MacSeriesExp5 = MathTex(r"\sum_{n=0}^\infty \frac{(i\theta)^n}{n!}",r"=",r"1",r"+ i\theta",r"- \frac{(\theta)^2}{2!}  ",r"\\- \frac{i(\theta)^3}{3!}",r"+ \frac{(\theta)^4}{4!}",r"+ \frac{(i\theta)^5}{5!} + ... "  ,
                     font_size = 90  )
        MacSeriesExp6 = MathTex(r"\sum_{n=0}^\infty \frac{(i\theta)^n}{n!}",r"=",r"1",r"+ i\theta",r"- \frac{(\theta)^2}{2!}  ",r"\\- \frac{i(\theta)^3}{3!}",r"+ \frac{(\theta)^4}{4!}",r"+ \frac{i(\theta)^5}{5!} + ... "  ,
                    font_size = 90  )

        self.play(ReplacementTransform(MacSeriesExp,MacSeriesExp1))
        self.play(ReplacementTransform(MacSeriesExp1,MacSeriesExp2))
        self.play(ReplacementTransform(MacSeriesExp2,MacSeriesExp3))
        self.play(ReplacementTransform(MacSeriesExp3,MacSeriesExp4))
        self.play(ReplacementTransform(MacSeriesExp4,MacSeriesExp5))
        self.play(ReplacementTransform(MacSeriesExp5,MacSeriesExp6))
        self.wait(1)

       

        msrearrangedf = MathTex(r"\sum_{n=0}^\infty \frac{(i\theta)^n}{n!}",r"=",r"1 - \frac{(\theta)^2}{2!} + \frac{(\theta)^4}{4!}" ,r"\\+ i \theta + i \frac{\theta^3}{3!} + i \frac{\theta^5}{5!} + ...",
                    font_size = 90  )
        txt5 = Tex("Rearranging", font_size=80).next_to(msrearrangedf,UP)

        self.play(ReplacementTransform(MacSeriesExp6,msrearrangedf),FadeIn(txt5))
        self.wait(1)
        ## separate into two lines and say they are series

        SeriesR = MathTex(r"=",r"1 - \frac{(\theta)^2}{2!} + \frac{(\theta)^4}{4!} - \frac{(\theta)^6}{6!} + ...\\" ,r"+ i(\theta + \frac{\theta^3}{3!} + \frac{\theta^5}{5!} + \frac{\theta^7}{7!} + ...)",
                        font_size = 80)

        braceCos = (Brace(SeriesR[1],direction=UP))
        braceCosText = braceCos.get_text("Maclaurin Cosine Series Expanded" ).scale(1.75)
        braceSin = (Brace(SeriesR[2],direction=DOWN))
        braceSinText = braceSin.get_text("Maclaurin Sine Series Expanded").scale(1.75)        

        self.play(ReplacementTransform(msrearrangedf,SeriesR) ,GrowFromCenter(braceCos),FadeIn(braceCosText),GrowFromCenter(braceSin),FadeIn(braceSinText),FadeOut(txt5))
        self.wait(2)
        self.play(FadeOut(braceCos),FadeOut(braceSin),FadeOut(braceCosText),FadeOut(braceSinText))


        ## then make each line equal to sin and cosine mac series 

        CosSinMac = MathTex(r"e^{i\theta}",r" = ",r"\sum_{n=0}^\infty \frac{(-1)^n\theta^{2n}}{(2n)!}",r"+ \\i(\sum_{n=0}^\infty \frac{(-1)^n\theta^{2n+1}}{(2n+1)!})",
                        font_size = 100  )
        finalLine = MathTex(r"e^{i\theta}", r" = Cos(\theta) + iSin(\theta)",
                        font_size = 100  )
        
        braceCos2 = (Brace(CosSinMac[2],direction=UP))
        braceCosText2 = braceCos2.get_text("Summation formula for Cos" ).scale(1.75)
        braceSin2 = (Brace(CosSinMac[3],direction=DOWN))
        braceSinText2 = braceSin2.get_text("Summation formula for Sin").scale(1.75)


        self.play(ReplacementTransform(SeriesR,CosSinMac),GrowFromCenter(braceCos2),FadeIn(braceCosText2),GrowFromCenter(braceSin2),FadeIn(braceSinText2))
        self.wait(1)
        self.play(FadeOut(braceCos2),FadeOut(braceSin2),FadeOut(braceCosText2),FadeOut(braceSinText2))
        


        ## then finally equal to sine and cosine of i0 then equal to e to i0

        self.play(ReplacementTransform(CosSinMac,finalLine))
        self.wait(2)
