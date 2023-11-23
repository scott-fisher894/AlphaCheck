
import pygame
from Board import *


class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT + 50))
        pygame.display.set_caption('AlphaCheck')
        self.game = Game()
        self.font = pygame.font.Font(None, 36)  # Choose appropriate font

    def draw_text_box(self, text):
        text_box_rect = pygame.Rect(0, HEIGHT, WIDTH, 50)  # Position and size of the text box
        self.screen.fill((0, 0, 0), text_box_rect)  # Fill with background color
        text_surface = self.font.render(text, True, (255, 255, 255))  # Render the text
        self.screen.blit(text_surface, (10, HEIGHT + 10))  # Adjust text position as needed

    def mainloop(self):
        screen = self.screen
        running = True

        while running:
            self.game.show_board(screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.game.handle_mouse_event(event)
            
            self.game.show_board(screen)
            status_text = self.game.get_status_text()
            self.draw_text_box(status_text)
            pygame.display.update()

    
main = Main()
main.mainloop()