#include "draw.h"
#include <iostream>
#include <cmath>


Grid createGrid() {
    return Grid(GRID_SIZE, std::vector<int>(GRID_SIZE, 0));
}

void clearGrid(Grid& grid) {
    for (auto& row : grid)
        std::fill(row.begin(), row.end(), 0);
}

void setCell(Grid& grid, int row, int col) {
    if (row >= 0 && row < GRID_SIZE && col >= 0 && col < GRID_SIZE)
        grid[row][col] = 1;
}

std::vector<int> flatten(const Grid& grid) {
    std::vector<int> flat;
    flat.reserve(GRID_SIZE * GRID_SIZE);
    for (int r = 0; r < GRID_SIZE; ++r)
        for (int c = 0; c < GRID_SIZE; ++c)
            flat.push_back(grid[r][c]);
    return flat;
}

void printFlattened(const Grid& grid) {
    std::cout << "[";
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (r || c) std::cout << ", ";
            std::cout << grid[r][c];
        }
    }
    std::cout << "]\n" << std::flush;
}


void drawGrid(sf::RenderWindow& window, const Grid& grid) {
    sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));

    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            cell.setPosition(c * CELL_SIZE, r * CELL_SIZE);
            cell.setFillColor(grid[r][c] ? sf::Color::Black : sf::Color::White);
            cell.setOutlineColor(sf::Color(220, 220, 220));
            cell.setOutlineThickness(-1.f);
            window.draw(cell);
        }
    }
}

bool pixelToGrid(const sf::Vector2i& px, int& row, int& col) {
    col = static_cast<int>(px.x / CELL_SIZE);
    row = static_cast<int>(px.y / CELL_SIZE);
    return (row >= 0 && row < GRID_SIZE && col >= 0 && col < GRID_SIZE);
}

void drawLine(Grid& grid, int r0, int c0, int r1, int c1) {
    int dr = std::abs(r1 - r0), dc = std::abs(c1 - c0);
    int sr = (r0 < r1) ? 1 : -1;
    int sc = (c0 < c1) ? 1 : -1;
    int err = dr - dc;

    while (true) {
        setCell(grid, r0, c0);
        if (r0 == r1 && c0 == c1) break;
        int e2 = 2 * err;
        if (e2 > -dc) { err -= dc; r0 += sr; }
        if (e2 <  dr) { err += dr; c0 += sc; }
    }
}