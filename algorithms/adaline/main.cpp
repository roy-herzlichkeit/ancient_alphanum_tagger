#include "adaline.h"
#include "../../common/draw.h"
#include <iostream>

static vector<int> gridToInput(const Grid& grid) {
    vector<int> input(ADALINE_INPUT_SIZE);
    for (int r = 0; r < GRID_SIZE; r++)
        for (int c = 0; c < GRID_SIZE; c++)
            input[r * GRID_SIZE + c] = grid[r][c] ? 1 : -1;
    input[ADALINE_PIXELS] = 1;
    return input;
}

int main() {
    cout << "=== ADALINE Character Recognition ===\n";
    
    vector<vector<double>> W(ADALINE_NUM_CLASSES, vector<double>(ADALINE_INPUT_SIZE, 0.0));
    double threshold, learning_rate, tolerance;
    cout << "Enter threshold: ";
    cin >> threshold;
    cout << "Enter learning rate (0-1): ";
    cin >> learning_rate;
    cout << "Enter tolerance: ";
    cin >> tolerance;
    adaline_train(W, "data/training.txt", threshold, learning_rate, tolerance);
    
    sf::RenderWindow window(
        sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE),
        "ADALINE Draw Pad  |  LMB: draw  |  Enter: classify  |  C: clear",
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
                    int cls = adaline_classify(X, W);
                    cout << "Detected: " << adaline_classLabel(cls) << endl;
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