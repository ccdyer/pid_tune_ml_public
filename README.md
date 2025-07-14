This is a utility for PID tuning using machine learning.
Currently, it is assumed that you have decent knowledge of python and PID tuning to use this utility.
Once you have installed the required python packages, run the program.
Generate a step change in your process, it should be within your normal operating range, and large enough to see a noticeable(>10%) change in your PV
Enter the CV Start, CV Final, PV Start, and PV Final values.  It is also recommended that you enter the min/max values for the CV and PV
Enter in your dead time, which is the amount of time it takes between changing your CV and seeing a change in the PV
Enter the time that the PV reached the 63.2% value
Enter your PID update rate in ms
Hit the Optimize PID button.  It will now run through (potentially) thousands of combinations to find the "ideal" PID tunings.  
Once complete, you can see the PID tuning parameters along with the overshoot and settling time values.
You can use the manual simulation button to get an estimate of how the PID will perform at different setpoints.
The manual simulation uses the CV and PV start values, and will try to reach the setpoint.

This assumes an ideal system, and is not perfect, but hopefully it will give you a good starting point for your tunings.

Please reach out and let me know how it works, or if you run into any issues.
