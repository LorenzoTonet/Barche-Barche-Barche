import pygame as pg

from ester_egg.source_code.boat import Boat
from ester_egg.source_code.utils import load_image, circle_rect_collision
from ester_egg.source_code.map import Goal
from ester_egg.source_code.settings import *

def main():
    pg.init()
    clock = pg.time.Clock()
    screen = pg.display.set_mode((WIDTH, HEIGHT))

    background = load_image("source_code/Assets/sea.jpeg")
    background = pg.transform.scale2x(background)

    p = Boat(load_image("source_code/Assets/barca.png"), 10, 2)
    pg.display.set_caption("Move the oligarca's boat into molo audace!")

    goal = Goal(800, 800, 213, 341, "source_code/Assets/molo audace.png")

    font = pg.font.SysFont(None, 72)
    win_text = font.render("OLIGARCA is in molo audace", True, (255, 255, 0))

    game_won = False
    while True:
        screen.blit(background, (0, 0))
        screen.blit(goal.image, goal.rect)


        if game_won == False:
            keys = pg.key.get_pressed()
            if keys[pg.K_UP]:
                p.move(up=True)
            if keys[pg.K_DOWN]:
                p.move(down=True)
            if keys[pg.K_LEFT]:
                p.move(left=True)
            if keys[pg.K_RIGHT]:
                p.move(right=True)
            p.update()

        rotated_sprite, boat_rect = p.get_rotated_sprite()
        screen.blit(rotated_sprite, boat_rect)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return

        if circle_rect_collision(p.pos, p.radius, goal.rect):
            game_won = True

        
        if game_won:
            screen.blit(win_text, (WIDTH // 2 - 500, HEIGHT // 2))
        clock.tick(60)
        pg.display.update()



if __name__ == "__main__":
    main()
    pg.quit()