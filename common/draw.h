#ifndef DRAW_H
#define DRAW_H

#include <SFML/Graphics.hpp>
#include <vector>

const int GRID_SIZE   = 7;
const int WINDOW_SIZE = 490;
const float CELL_SIZE = static_cast<float>(WINDOW_SIZE) / GRID_SIZE;

using Grid = std::vector<std::vector<int>>;

Grid  createGrid();
void  clearGrid(Grid& grid);
void  setCell(Grid& grid, int row, int col);

std::vector<int> flatten(const Grid& grid);

void  printFlattened(const Grid& grid);

void  drawGrid(sf::RenderWindow& window, const Grid& grid);

bool  pixelToGrid(const sf::Vector2i& px, int& row, int& col);
void  drawLine(Grid& grid, int r0, int c0, int r1, int c1);

#endif // DRAW_H
