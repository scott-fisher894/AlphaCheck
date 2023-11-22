
import pygame
import sys

from Board import *


class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('AlphaCheck')
        self.game = Game()

    def mainloop(self):

        screen = self.screen
        running = True

        while running:
            self.game.show_board(screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
            
            
            pygame.display.update()

    
main = Main()
main.mainloop()