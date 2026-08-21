import pygame as pg
import os
from source_code.settings import BOAT_START_X, BOAT_START_Y, SEA_FRICTION

class Boat:
    def __init__(self, sprite, height, acceleration):
        self.sprite = sprite
        self.height = height

        self.pos = pg.math.Vector2(
            BOAT_START_X,
            BOAT_START_Y
        )

        self.velocity = pg.math.Vector2(0, 0)

        self.acceleration = acceleration
        self.max_speed = acceleration * 5

        self.friction = SEA_FRICTION

        self.angle = 0
        self.turn_speed = 3.0

        self.rect = self.sprite.get_rect(center=self.pos)
        self.radius = 100

    def move(self, up=False, down=False, left=False, right=False):
        forward = pg.math.Vector2(1, 0).rotate(-self.angle)

        if up:
            self.velocity += forward * self.acceleration
        if down:
            self.velocity -= forward * self.acceleration * 0.5

        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)

        speed = self.velocity.length()
        turn = self.turn_speed * (speed / self.max_speed)
        if left:
            self.angle += turn
        if right:
            self.angle -= turn

    def update(self):
        self.velocity *= self.friction
        self.pos += self.velocity
        self.rect.center = self.pos

    def get_rotated_sprite(self):
        rotated = pg.transform.rotate(self.sprite, self.angle)
        rect = rotated.get_rect(center=self.pos)
        self.rect = rect
        return rotated, rect

    def get_state(self):
        return {
            "position": (self.pos.x, self.pos.y),
            "velocity": (self.velocity.x, self.velocity.y),
            "angle": self.angle
        }