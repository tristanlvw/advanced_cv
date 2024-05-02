Because my code for each question is quite long, I have supplied 3 jupyter notebooks for Q1, Q2.1, Q2.2 in the root directory:

- Q1.ipynb contains the code for questions 1.1, 1.2, 1.3
- Q2_dclgan.ipynb contains the code for 2.1 which runs the DCLGAN model
- Q2_mcl_dclgan.ipynb contains the code for 2.2 which runs my improved MCL-DCLGAN model

All of my functions for Q1.ipynb are defined in the 'dataset' directory. My functions for Q2p1.ipynb, Q2p2.ipynb are defined in python files in the 'model' directory.

My multimedia files (image samples) for questions 1.1, 1.2 and 1.3 are stored in Q1.1_samples, Q1.2_samples, Q1.3_samples. The contents of Q1.2_samples are further organized into subdirectories for each classes.

In the root directory, I also supply:

- my report report_qpnz47.pdf
- my video from question 2.1: test_dclgan.mp4
- my video from question 2.2: test_mcl_dclgan.mp4

To save space and clutter in my submission, I have not included my frame or patch dataset files (outside of those specifically requested in the brief), or in some cases the necessary subdirectory folders. To run certain cells, these directories will need to be created. This will be clear based on the code in that particular cell. 

E.g. for DCLGAN and MCL-DCLGAN training, we need a dataset folders in a directory 'data' which is one level above the submission folder and contains the following subdirectories:

    ../data
    
        --> /frame_dataset
            --> trainA
            --> trainB
        
        --> /classified
        
            --> game
                --> head_and_shoulders_front
                --> head_and_shoulders_back
                --> full_front
                --> full_back
                --> other
                
            --> movie
                --> head_and_shoulders_front
                --> head_and_shoulders_back
                --> full_front
                --> full_back
                --> other
        