# ImageMorphing
Morph face images from different viewpoint to produce enhanced face

This tool requires a set of five images to be stored in the folder.
The image is assumed to be of resolution 300x300

#Usage:
To run the tool,
python img_morph.py /path_to_txt_file_containing_the_five_images.txt

An example set of images are there in "face_db_1" and "face_db_2"
The txt file indicating their location is inputs_1.txt and inputs_2.txt

#Example
python -i img_morph.py -i inputs_1.txt
python -i img_morph.py -i inputs_2.txt

#Result:
A window with the morphed face will be displayed.
