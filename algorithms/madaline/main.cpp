// main.cpp — MADALINE main program (stub)
#include "madaline.h"
#include "../../common/draw.h"
#include <iostream>

static vector<int> gridToInput(const Grid& grid) {
    vector<int> input(MADALINE_INPUT_SIZE);
    for (int r = 0; r < GRID_SIZE; r++)
        for (int c = 0; c < GRID_SIZE; c++)
            input[r * GRID_SIZE + c] = grid[r][c] ? 1 : -1;
    input[MADALINE_PIXELS] = 1;
    return input;
}

int main() {
    cout << "=== MADALINE Character Recognition ===\n";
    cout << "TODO: Implement MADALINE algorithm\n";
    
    Madaline network;
    madaline_train(network, "data/training.txt");
    
    sf::RenderWindow window(
        sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE),
        "MADALINE Draw Pad - TO BE IMPLEMENTED", 
        sf::Style::Titlebar | sf::Style::Close
    );
    window.setFramerateLimit(60);

    Grid grid = createGrid();
    bool drawing = false;
    int prevRow = -1, prevCol = -1;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Enter) {
                    vector<int> X = gridToInput(grid);
                    int cls = madaline_classify(X, network);
                    cout << "Detected: " << madaline_classLabel(cls) << endl;
                }
                if (event.key.code == sf::Keyboard::C) {
                    clearGrid(grid);
                }
            }

            if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                drawing = true;
                int r, c;
                if (pixelToGrid(sf::Vector2i(event.mouseButton.x,
                                             event.mouseButton.y), r, c)) {
                    setCell(grid, r, c);
                    prevRow = r;
                    prevCol = c;
                }
            }

            if (event.type == sf::Event::MouseButtonReleased &&
                event.mouseButton.button == sf::Mouse::Left) {
                drawing = false;
                prevRow = prevCol = -1;
            }

            if (event.type == sf::Event::MouseMoved && drawing) {
                int r, c;
                if (pixelToGrid(sf::Vector2i(event.mouseMove.x,
                                             event.mouseMove.y), r, c)) {
                    if (prevRow >= 0 && prevCol >= 0)
                        drawLine(grid, prevRow, prevCol, r, c);
                    else
                        setCell(grid, r, c);
                    prevRow = r;
                    prevCol = c;
                }
            }
        }

        window.clear(sf::Color::White);
        drawGrid(window, grid);
        window.display();
    }

    return 0;
}