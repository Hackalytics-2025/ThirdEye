## Inspiration

ThirdEye was born out of a desire to empower the visually impaired with a tool that enhances situational awareness in real time. Inspired by the challenges faced by the visually impaired, our team aimed to create a system that not only detects objects but also interprets complex environmental cues—transforming visual data into meaningful audio feedback. We were driven by the potential of integrating modern AI and computer vision techniques to improve everyday life and increase independence for users.

## What it does

ThirdEye is an intelligent camera system that actively assists visually impaired users using multiple modes. The modes are listed below:

**Explore Mode (when user is interacting with the outside world):**

_Object Detection & Proximity Alerts:_ Utilising depth recognition (via MiDaS) to identify objects in the live camera feed and warn users if an object is too close. Example: warns user walking on the road, if there's an obstacle like a lamp post in the way.

_Optical Character Recognition (OCR):_ Recognising text within the environment, enabling the system to read out important information. Example: reading traffic signs

_Weather Updates:_ Integrating with a weather API (powered by Google Gemini) to provide current conditions and suggest lifestyle adjustments based on the forecast. Example: On a rainy day, audio cues reference the weather outside and advice user to carry an umbrella.

Data Visualisation: Based on the depth data gathered from MiDaS it overlays a heat map on the video feed to visualise the dynamic depth scores it actively assigns to objects in the surroundings.

**Conversational Mode (for one-on-one conversations):**

_Emotional Recognition:_ Detecting and interpreting the emotional state of individuals in view using facial expression recognition (via FER). The aim is to enhance the quality of social interactions the user has by providing audio cues on the mood of the person in view.

## How we built it

First, we used OpenCV to capture video and process each frame. We then brought in PyTorch to handle model inference. Our next step was integrating the pre-trained models. For object detection, we selected YOLOv5. This model processes each frame, drawing bounding boxes and calculating confidence scores. We adjusted these detections based on area and proximity to ensure we only alerted the user to important objects. **For depth estimation, we chose MiDaS. We learned to transform each frame into a tensor, run the model, and then resize the output to match the original frame. We used MiDaS to implement data analytics techniques like Convolutional Neural Networks (Uses CNNs to extract hierarchical features from input images) and Regression Techniques(Implements regression to predict continuous depth values for each pixel in the image).**

We then added a text-to-image component using the BLIP captioning model from Salesforce. This model helped us generate a description of what the camera saw. In parallel, we integrated an OCR tool to extract and read any text from the environment. For the emotion detection part, we used a facial emotion recognition model. After YOLO detected a face, this module analysed the facial expression to gauge the emotion.

The system was designed to run multiple tasks at the same time. We set up separate threads: one that periodically generated audio captions (the captioning loop) and another that constantly monitored depth data to trigger alerts when objects were too close (the depth alert loop). To generate the audio feedback, we used gTTS and played the sound using Pygame’s mixer.

Lastly, we incorporated weather data using the Visual Crossing Weather API that we fed to Google’s Gemini. This API fetched current weather conditions, and we then converted that data into natural language advisories, which were also turned into speech.

## Challenges we ran into
Building ThirdEye was as complicated as we thought it would be. The main challenge we found was combining multiple AI models (each with unique input/output requirements) into a seamless pipeline proved challenging. Finding and integrating the MiDaS model proved to be the biggest obstacle we overcame as it wasn't compatible with data analytics we wanted to run on the depth of the objects to generate the data visualisation in the form of heat maps. Then, once we were able to generate a live heat map on the video feed we encountered the extensive GPU requirements for the project. To ensure minimal latency for time-sensitive audio feedback, we optimised the software for efficient GPU memory utilisation and constrained all integrated AI models within the GPU's capacity. Balancing the sensitivity of object detection, OCR, and facial recognition while minimising false positives and negatives were one of the many other minor errors we dealt with. By tweaking confidence thresholds, trying out different facial expression datasets, we were eventually able to make a reliable project.

## Accomplishments that we're proud of

Solving the biggest challenge we faced was the biggest achievement for us as a team. Successfully combining object detection, text recognition, and emotion analysis into a single, coherent system was essential in making this project achieve the desired task. We are proud of the extensive complimentary features we were able to integrate into the project to make it a well-rounded project. Creating a system that actively warns users about obstacles, analysing conversations in real life, and giving weather data inputs makes it a more complete product in terms of solving the day-to-day struggles faced by the visually impaired. To further enhance system performance and reliability, we implemented a modular architecture that allows each AI component to function autonomously while efficiently sharing data through optimised GPU-based processing. Our object detection module, powered by YOLO, was fine-tuned to maintain accuracy across different environments (indoors and outdoors), while our OCR component was integrated to reliably extract and process textual information in real time (for eg. reading traffic signs). Additionally, the emotion analysis module leverages state-of-the-art FER techniques to provide precise evaluations of facial expressions. These targeted implementations directly address our initial development challenges and set a solid foundation for future scalability and feature integration.

## What we learned

We learned that integrating different AI models can be tricky. Each model had its own input and output formats, so getting them to work together required a lot of tweaking. Adapting the MiDaS model for our heat maps taught us how to modify existing architectures to fit our needs. We also discovered how important GPU memory management is. Optimising resource allocation and adjusting model parameters helped us maintain real-time performance. Balancing sensitivity for object detection, OCR, and facial recognition was another key insight. We had to fine-tune thresholds and choose the right datasets to reduce false positives and negatives. Overall, these lessons have given us a solid foundation for future improvements. We now have a better grasp of both the technical challenges and the practical steps needed to create reliable assistive technology.

## What's next for ThirdEye

Since the inception of our idea, there is one thing we yearned for: Lidar (Light Detection and Ranging). This would enable us to get the most accurate depth perception and thus, create the best platform for developing a data analytics model that generates relevant audio feedback. However, this would make a lot of our current code pertaining to depth perception in reference to the camera frame useless :( But if our end goal is building on product viability and reliability, Lidar is the future for us.

Additionally, we had the idea of incorporating GPS navigation to guide users with step-by-step navigation and provide alerts about crowded areas or heavy traffic (using existing depth analysis). There is also work needed to be done to expand the range of detectable emotions and refine the accuracy of facial expression analysis that still gets some expressions wrong.

To expand on making the product truly inclusive and global, we could develop multi-language audio outputs to cater to an international user base, making the system accessible to more people.
