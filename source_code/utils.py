import pygame as pg
import os

# quick function to load an image
def load_image(path):
    return pg.image.load(path).convert_alpha()

def circle_rect_collision(circle_pos, radius, rect):
    closest_x = max(rect.left, min(circle_pos.x, rect.right))
    closest_y = max(rect.top, min(circle_pos.y, rect.bottom))
    distance = circle_pos.distance_to((closest_x, closest_y))
    return distance < radius