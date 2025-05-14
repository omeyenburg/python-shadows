# -*- coding: utf-8 -*-
from shadow import find_corners, get_edges, get_triangle_points
import pygame
import numpy


BLOCKSIZE = 64
WALLS = numpy.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=int,
)


def coord_to_pixel(x, y):
    return (x + 1) * BLOCKSIZE, (y + 1) * BLOCKSIZE


def pixel_to_coord(x, y):
    return x / BLOCKSIZE - 1, y / BLOCKSIZE - 1


def draw_shadows(window, view):
    # Get corners and edges
    corners, additional_corners = find_corners(view)
    edges = get_edges(corners)
    corners = corners.union(additional_corners)

    # Use mouse position as light source
    # light_sources = numpy.argwhere(view == 2)
    mousePos = pixel_to_coord(*pygame.mouse.get_pos())
    light_sources = [mousePos]

    # Sort and draw corners
    for light_source in light_sources:
        corners = list(additional_corners.union(corners))

        triangle_points = get_triangle_points(
            view, light_source, numpy.array(corners), edges
        )
        triangle_points.sort(key=lambda n: n[0], reverse=False)

        # Draw to shadow surface
        triangle_points = tuple(map(lambda n: coord_to_pixel(*n[1:]), triangle_points))
        if len(triangle_points) > 2:
            pygame.draw.polygon(window, (200, 100, 250), triangle_points)

        # Draw corner rays
        for corner in triangle_points:
            pygame.draw.line(window, (255, 255, 255), coord_to_pixel(*mousePos), corner)


def main():
    pygame.init()
    pygame.display.set_caption("Shadows")

    window_size = (WALLS.shape[0] * BLOCKSIZE, WALLS.shape[1] * BLOCKSIZE)
    window = pygame.display.set_mode(window_size)

    while True:
        window.fill((0, 0, 0))
        mousePos = (
            pygame.mouse.get_pos()[0] / BLOCKSIZE,
            pygame.mouse.get_pos()[1] / BLOCKSIZE,
        )

        for x, y in numpy.ndindex(WALLS.shape):
            if WALLS[x, y]:
                pygame.draw.rect(
                    window,
                    (255, 255, 255),
                    (x * BLOCKSIZE, y * BLOCKSIZE, BLOCKSIZE, BLOCKSIZE),
                )

        if sum(mousePos) > 0:
            draw_shadows(window, WALLS)

        pygame.display.flip()
        if pygame.event.get(pygame.QUIT):
            pygame.quit()
            return


if __name__ == "__main__":
    main()
