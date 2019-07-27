# Music Generation using AI
## Generating Music from training a system for specific instruments

***********************************************************************************************************************
### *Technology stack*:

Python, Tensorflow, Keras
Jupyter Notebook (we have used here ipynb jupyter notebooks here)

***********************************************************************************************************************
### *Files*:
```
   * Model Folder

	|->Generate_Music.ipynb         #file to generate music

	|->FinalTraining.ipynb 	#file to train the model

	|->sequence_models.py 		# all utility functions grouped together

	|->Data Folder 			#folder for first training file

		|->Data_Tunes.txt #contains data of music in ABC notation form

		|->Model_Weights  	#consists of model weights at specific intervals

		|->char_to_index.json  	#stores character to index dictionary form for mapping characters

	|->Data2 Folder #folder for second training file

		|->Model_Weights  	#consists of model weights at specific intervals

		|->char_to_index.json 	#stores character to index dictionary form for mapping characters

```

***********************************************************************************************************************

### *Details*:
```
1. The model here creates a 5 layer neural network of LSTMS with sequntial models to help computer learn to generate
   music based on training from 258 music ABC notation data.
2. Here, with each character as a input the model tries to predict a new character.
3. Starting input has to be given to start from.
4. Details are written in the ipynb files also for specific functions.
5. Here only one instrument music files has been given, this can be increased to multiple music instruments in future
   approach.
```
***********************************************************************************************************************

### *Running the code*:

```
1. Open Model folder
2. Run all the cells in Generate_Music.ipynb
3. Enter the required inputs.
4. Copy the output from cell and go this link to convert this notation to music midi file https://colinhume.com/music.aspx
5. i) Paste the copied generated output to this website editor.
   ii) Go to Data_Tunes.txt
   iii) To convert we need to specify X and T compulsorily so, you can copy this from the input file from any music data
       and use that.
       for example:(Paste this ABOVE the generated output in the website editor)
        X: 6
	T:Arthur Darley
	% Nottingham Music Database
	S:Formally, but with lift, via Phil Rowe
   iv) Now press the Convert button.
   v) Press Play button and you will be asked to download the file, download the file and play it in Windows Media Player.

```
***********************************************************************************************************************
### *Authors*:

- Sushant Kumar   (Roll No. 85) Email ID: kumarsr_1@rknec.edu
- Sarthak Baiswar (Roll No. 76) Email ID: baiswarsp@rknec.edu

***********************************************************************************************************************