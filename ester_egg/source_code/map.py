import pygame as pg

class Goal():
    def __init__(self, x, y, width, height, image_path):
        self.rect = pg.Rect(x, y, width, height)
        self.image = pg.image.load(image_path).convert_alpha()

