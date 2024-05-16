# Violence Detection and Person Identification

This project uses a combination of object detection and person re-identification (re-id) to detect violence and identify persons in images.

## Project Structure

### Key Files

- `app_demo.py`: This is the main application file. It uses Streamlit to create a web application where users can upload images for violence detection and person identification.
- `detection.py`: This file contains the violence detection logic. It uses the YOLO model for object detection and calculates Intersection over Union (IOU) to filter bounding boxes.
- `re_id.py`: This file contains the person re-identification logic.
- `model/best_violence_det_02.pt`: This is the pre-trained model used for violence detection.
- `model/ft_ResNet50/`: This directory contains the files related to the ft_ResNet50 model used for person re-identification.

## Usage

To run the application, execute the following command:

```bash
streamlit run app_demo.py
