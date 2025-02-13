# VIDEODRAW

## Dependencies
- python3
- pip
- install requirements.txt via pip (pip install -r requirements.txt)

## How to start
- To run app, python3 interface/app.py
- in general, every executable needs to be executed from root

## Interface
The Hand recognition Model is based on Kazuhito Takahashis Model https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe.git and was modified. <br>
Drawing Interface for User with 4 possible hand gestures:

1. Neutral <br>
![NeutralPose](./gesture-pictures/Neutral.png) <br>
Neutral gesture which doesnt affect Drawing <br>

2. Draw <br>
![DrawPose](./gesture-pictures/Draw.png) <br>
Gesture for drawing, draws at the position of the index finger <br>

3. DeleteLast <br>
![DeleteLastPose](./gesture-pictures/DeleteLast.png) <br>
Deletes last annotation of Drawing <br>

4. DeleteAll <br>
![DeleteAllPose](./gesture-pictures/DeleteAll.png) <br>
Deletes whole Drawing <br>

## Custom Aliases for frequent use
alias cdvdraw='cd {your_path_to_videodraw}' <br>
alias activate='source venv/bin/activate' <br>
alias vdraw='python interface/app.py' <br>
alias vdrawinit='cdvdraw activate vdraw' <br>


