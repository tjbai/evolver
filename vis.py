from manim import *
from collections import defaultdict

class Evolver(Scene):
    def construct(self):
        self.encoder_circles = [[self.create_circle(i, j) for j in range(4)] for i in range(4)]
        encoder_circles_group = VGroup(*[circle for row in self.encoder_circles for circle in row])
        
        self.encoder_lines = self.create_bidirectional_lines()
        encoder_lines_group = VGroup(*[line for layer in self.encoder_lines.values() for line in layer])
        
        encoder = VGroup(encoder_lines_group, encoder_circles_group)
        encoder.shift(LEFT * 4)
        
        input_labels = VGroup(*[Text(f"Node {j+1}", font_size=24).next_to(self.encoder_circles[3][j], DOWN) for j in range(4)])
        
        self.play(FadeIn(encoder), Write(input_labels))
        self.wait(1)
        
        for row in range(3, -1, -1):
            self.light_up_row(row)
            self.wait(0.5)
      
        output_labels = VGroup(*[
            Text(f'Memory {j+1}', font_size=24).next_to(self.encoder_circles[0][j], UP)
            for j in range(4)
        ])

        self.play(Write(output_labels))
        self.wait(1)
        
        self.decoder_circles = [[self.create_circle(i, j) for j in range(4)] for i in range(4)]
        decoder_circle_group = VGroup(*[circle for row in self.decoder_circles for circle in row])
       
        self.incoming = defaultdict(list) 
        self.decoder_lines = self.create_causal_lines()
        decoder_lines_group = VGroup(*[line for layer in self.decoder_lines.values() for line in layer])
        
        decoder = VGroup(decoder_lines_group, decoder_circle_group)
        decoder.shift(RIGHT * 4)
        
        self.play(FadeIn(decoder))
        self.wait(1)
        
        memory = self.encoder_circles[0]
        
        for i in range(4):
            for j in range(3, -1, -1):
               cur = self.decoder_circles[j][i]
               lines = self.incoming[(j+1, i)]
               causal_attention = [line.animate.set_color(GREEN) for line in lines]
               cross_attention_lines = VGroup(*[Line(node, cur, color=GREEN) for node in memory])
               self.play(*causal_attention, FadeIn(cross_attention_lines), run_time=0.5)
               self.play(cur.animate.set_color(GREEN).set_fill(GREEN, opacity=1), FadeOut(cross_attention_lines), run_time=0.5)
               
            output_label = VGroup(Text(f'Edit {i}', font_size=24).next_to(self.decoder_circles[0][i], UP))
            self.play(Write(output_label))
            self.wait(0.5)
            
            target_label = VGroup(Text(f'Target {i}', font_size=24).next_to(self.decoder_circles[3][i], DOWN))
    
    def create_circle(self, i, j):
        circle = Circle(radius=0.3, color=WHITE).set_fill(WHITE, opacity=1)
        circle.move_to([j*1.5-2.25, -i*1.5+2.25, 0])
        return circle
    
    def create_bidirectional_lines(self):
        lines = {1: [], 2: [], 3: []}
        for i in range(1, 4):
            for j in range(4):
                for k in range(4):
                    end = self.encoder_circles[i-1][k].get_center()
                    start = self.encoder_circles[i][j].get_center()
                    line = Line(start, end, color=WHITE)
                    lines[i].append(line)
        return lines
    
    def create_causal_lines(self):
        lines = {1: [], 2: [], 3: []}
        for i in range(1, 4):
            for j in range(4):
                for k in range(j+1):
                    end = self.decoder_circles[i-1][j].get_center()
                    start = self.decoder_circles[i][k].get_center()
                    line = Line(start, end, color=WHITE)
                    self.incoming[(i, j)].append(line)
                    lines[i].append(line)
        return lines
    
    def light_up_row(self, row):
        row_circles = self.encoder_circles[row]
        row_lines = self.encoder_lines.get(row+1, [])
        if row_lines: self.play(*[line.animate.set_color(YELLOW) for line in row_lines], run_time=0.5)
        self.play(*[circle.animate.set_color(YELLOW).set_fill(YELLOW, opacity=1) for circle in row_circles], run_time=0.5)