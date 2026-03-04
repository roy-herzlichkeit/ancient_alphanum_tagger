#include "mlp.h" 
#include "../../common/draw.h"

static vector<int> gridToInput(const Grid& grid) {
    vector<int> input(MLP_INPUT_SIZE);
    for (int r = 0; r < GRID_SIZE; r++)
        for (int c = 0; c < GRID_SIZE; c++)
            input[r * GRID_SIZE + c] = grid[r][c] ? 1 : -1;
    input[MLP_PIXELS] = 1;
    return input;
}

int main() {
    double learning_rate;
    int max_epochs;
    cout << "=== MLP Character Recognition ===\n";
    cout << "Enter the Learning rate (0 - 1): ";
    cin >> learning_rate;
    cout << "Enter total Epochs to be performed (1 - 2000): ";
    cin >> max_epochs;
    
    MLP network;
    mlp_train(network, "data/training.txt", learning_rate, max_epochs);
    
    sf::RenderWindow window(
        sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE),
        "MLP Draw Pad  |  LMB: draw  |  Enter: classify  |  C: clear",
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
                    int cls = mlp_classify(X, network);
                    cout << "Detected: " << mlp_classLabel(cls) << endl;
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