from manim import *

class TransformerAnimation(Scene):
    def construct(self):
        # Create the main components
        input_text = Text("Input").scale(0.7)
        embedding = Rectangle(height=2, width=1).set_fill(BLUE, opacity=0.5)
        embedding_text = Text("Embedding").scale(0.5).next_to(embedding, DOWN)
        
        positional_encoding = Rectangle(height=2, width=1).set_fill(GREEN, opacity=0.5)
        pe_text = Text("Positional\nEncoding").scale(0.5).next_to(positional_encoding, DOWN)
        
        attention = Rectangle(height=2, width=2).set_fill(RED, opacity=0.5)
        attention_text = Text("Self-Attention").scale(0.5).next_to(attention, DOWN)
        
        ffn = Rectangle(height=2, width=2).set_fill(YELLOW, opacity=0.5)
        ffn_text = Text("Feed Forward\nNetwork").scale(0.5).next_to(ffn, DOWN)
        
        output = Text("Output").scale(0.7)

        # Position the components
        components = VGroup(input_text, embedding, positional_encoding, attention, ffn, output)
        components.arrange(RIGHT, buff=1)
        
        # Add labels
        labels = VGroup(embedding_text, pe_text, attention_text, ffn_text)

        # Create arrows
        arrows = VGroup()
        for i in range(len(components) - 1):
            arrow = Arrow(components[i].get_right(), components[i+1].get_left(), buff=0.1)
            arrows.add(arrow)

        # Animation
        self.play(Write(input_text))
        self.wait(0.5)
        
        self.play(Create(embedding), Write(embedding_text))
        self.play(GrowArrow(arrows[0]))
        self.wait(0.5)
        
        self.play(Create(positional_encoding), Write(pe_text))
        self.play(GrowArrow(arrows[1]))
        self.wait(0.5)
        
        self.play(Create(attention), Write(attention_text))
        self.play(GrowArrow(arrows[2]))
        self.wait(0.5)
        
        self.play(Create(ffn), Write(ffn_text))
        self.play(GrowArrow(arrows[3]))
        self.wait(0.5)
        
        self.play(Write(output))
        self.wait(1)

        # Emphasize the flow
        self.play(
            components.animate.set_color(GRAY),
            labels.animate.set_color(GRAY),
            arrows.animate.set_color(BLUE),
            run_time=2
        )
        
        for arrow in arrows:
            self.play(arrow.animate.set_color(YELLOW), run_time=0.5)
        
        self.wait(2)
        
