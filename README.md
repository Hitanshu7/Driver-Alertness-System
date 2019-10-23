#Winner of Smart-India-Hackathon-2019#The Game
Driver alertness and rash driving detection

This repository consists of different files and data for driver alertness system.
We have implemented different features to provide alertness to the driver at the time of driving.
This solution addresses seven different use cases in order to provide better feedback to the driver which will help in reducing the number of accidents.

Features:
1) Drowsiness detection
2) Alcohol detection
3) Emotion analysis and detection
4) Lane detection
5) Proximity sensing and audio alert(in different languages)
6) Traffic sign detection 
7) Distraction by the use of mobile
8) Reminder for maintenance

Drowsiness detection:
The system detects Eye Aspect Ratio(EAR) using Haar Cascade classifier along with Adaboost optimizer which is the world's finest optimizer for binary classification, unlike traditional approaches.

Alcohol detection:
The system uses MQ3 sensor and STM32 microcontroller to detect alcohol vapours in the driver's breath. An external switch is connected to control the motion of the servo motor. Servo motor is used to represent engine system in our prototype. When alcohol amount greater than the legal limit is detected, the car won't start and the driver will receive an LED indication.
 
Emotion Analysis and detection:
The dataset used to train this model is Kaggle-fer2013. The model detects various emotions of the driver such as, Happy, Sad, Angry, Surprised, etc. If the driver is angry and hence it’s not safe for him to drive, An SMS alert will be sent to his Guardian/Emergency contact. The Guardian/Emergency contact can communicate to the driver using our WebRtc and the contact would get the live videofeed of the driver and duplex audio communication between the driver and the contact is possible.

Lane detection:
The model detects the current lane of the vehicle and alerts the driver in case of any deviations. Rash driving can also be detected if too many lane changes occur within a particular time period. 
The system uses sobel and gradient filters for detection of the lines. Radius of curvature value is used to decide the threshold. Sliding window is used to detect lane changes. We are using three colour scheme to detect the rash driving cases:
Green- Normal driving, Blue- Threshold crossed once, Red- If the threshold is crossed continuously for next 50 frames- In this case we can send some alert signal.

Proximity sensing and audio alert:
This feature will keep the driver alert about the distance with objects and other cars avoiding crashes. It will give different audio warnings to the driver whenever a vehicle or pedestrian is detected in close vicinity. The audio output is available in different languages and the driver has a choice to select a language as per his preference. Proximity is calculated using just one webcam unlike traditional approaches which need the use of multiple cameras.

Traffic sign detection:
Traffic signs are sometimes ignored when driver isn't paying attention, this feature gives audio warning to the driver whenever any important traffic sign is detected. Currently the model is trained only for stop sign.

Distraction by the use of mobile:
EAR is used to implement this technique. if the driver is on his/her phone for a long time then the system sets off a buzzer sound which alerts the driver.

Reminder for maintenance:
This system uses a database which consists of driver's login credentials, basic information about vehicle and driver, EmergencyContact, Scheduled maintenance date. As per the scheduled date an alert is sent to the driver to remind about the maintenence. Once the vehicle maintenence is done, the datbase automatically updates the date for next cycle. Difference between two maintenance cycles is 35 days.
