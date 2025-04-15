# Chess Game with GUI

This is a chess game implementation with a graphical user interface using Python and Tkinter.

## Requirements

- Python 3.x
- Pillow (PIL) library
- Chess piece images

## Setup

1. Install the required Pillow library:
```bash
sudo apt-get install python3-pil.imagetk
pip install Pillow
```

2. Create a 'pieces' directory in the project folder
3. Add chess piece images to the 'pieces' directory with the following naming convention:
   - wPawn.png
   - wRook.png
   - wKnight.png
   - wBishop.png
   - wQueen.png
   - wKing.png
   - bPawn.png
   - bRook.png
   - bKnight.png
   - bBishop.png
   - bQueen.png
   - bKing.png

## Features

- Graphical chess board interface
- Human vs Computer gameplay
- Move history display
- Valid move highlighting
- Game state notifications
- Piece movement validation
- Check and checkmate detection

## How to Play

1. Run the game:
```bash
python chess_gui.py
```

2. Play as White:
   - Click on a piece to select it
   - Valid moves will be highlighted in green
   - Click on a highlighted square to make your move
   - The computer will automatically make its move as Black

3. The game ends when there is a checkmate or stalemate

## Game Controls

- Left-click: Select piece and make moves
- The game automatically handles turns between player and computer
- Move history is displayed on the right side of the board